import os
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from langchain.schema import HumanMessage, SystemMessage

from agents.meta_agents.initializer import initialize_agents
from agents.base import BaseAgent, LLMToolAgent, Portfolio, Input, Output, Scratchpad
from base.helper import TraceEvent, TracePersister
from base.logger import setup_logging

log = setup_logging()

# ---------------------------------------------------------------------------
# Plan & execution graph
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    source: str
    target: str

def topological_sort(agents, edges) -> List[str]:
        """
        Kahn algorithm implementation for topological sorting
        Returns topologically sorted list of nodes
        """
        graph: Dict[str, List[str]] = defaultdict(list)
        for e in edges:
            graph[e.source].append(e.target)
        # Calculate in-degree for each node
        in_degree = defaultdict(int)
        all_nodes = set()
        
        # Collect all nodes from edges
        for edge in edges:
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)
            in_degree[edge.target] += 1
        
        # Include all agents in the plan
        try:
            for node in agents.keys():
                all_nodes.add(node)
        except:
            all_nodes = set(agents)  # Handle case where agents is a list
        
        # Ensure all nodes are in in_degree dict
        for node in all_nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Find all nodes with in-degree 0 as starting points
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []
        
        while queue:
            # Take a node with in-degree 0
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degree of all adjacent nodes
            for target in graph.get(current, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
        
        # Check for cycles
        if len(result) != len(all_nodes):
            raise ValueError("DAG contains cycles!")
        
        return result


class Plan:
    def __init__(self, agents: Portfolio, edges: List[Edge]):
        self.agents = agents
        self.edges = edges
        self._graph: Dict[str, List[str]] = defaultdict(list)
        for e in edges:
            self._graph[e.source].append(e.target)
        log.debug("Plan created with nodes=%d, edges=%d", len(agents), len(edges))

    def list_strings(self) -> List[str]:
        """Return a flat list of all meaningful string artefacts.

        * agent names
        * agent `goal` descriptions
        * edge labels (source→target pairs as "source->target")
        """
        strings: List[str] = []
        # agent names and goals
        for name, agent in self.agents.items():
            strings.append(name)
            if hasattr(agent, "goal") and isinstance(agent.goal, str):
                strings.append(agent.goal)
        # edges
        strings.extend([f"{e.source}->{e.target}" for e in self.edges])
        return strings

    def execute_dfs(
        self,
        memory: Optional[Scratchpad] = None,
        tracer: Optional[TracePersister] = None,
        payload: Optional[Input] = None,
    ) -> Output:
        """Execute plan using DFS traversal (legacy method)"""
        log.info("Execute plan with DFS...")
        memory = memory or Scratchpad()
        out = payload
        for root in self._roots():
            out = self._dfs(root, out, memory, tracer)
        log.info("DFS execution complete")
        return out

    def _dfs(
        self,
        node: str,
        data: Input,
        memory: Scratchpad,
        tracer: Optional[TracePersister],
    ) -> Output:
        agent = self.agents[node]
        log.debug("DFS → %s", node)
        output = agent(memory)
        if tracer:
            tracer.save(TraceEvent(datetime.now(timezone.utc), node, data, output))
        for child in self._graph.get(node, []):
            output = self._dfs(child, output, memory, tracer)
        return output

    def _roots(self) -> List[str]:
        targets = {e.target for e in self.edges}
        sources = {e.source for e in self.edges}
        return list(sources - targets) or list(sources)

    def execute(
        self,
        memory: Optional[Scratchpad] = None,
        tracer: Optional[TracePersister] = None,
        payload: Optional[Input] = None,
    ) -> Output:
        """
        Execute plan using topological order by default
        This ensures each agent runs exactly once and all dependencies are satisfied
        Falls back to DFS execution if topological sort fails (e.g., due to cycles)
        """
        log.info("Execute plan with topological order...")
        memory = memory or Scratchpad()
        
        # Get topologically sorted node order
        try:
            topo_order = topological_sort(self.agents, self.edges)
        except ValueError as e:
            log.error("Failed to get topological order: %s", e)
            # Fallback to DFS execution
            log.info("Falling back to DFS execution")
            return self.execute_dfs(memory, tracer, payload)
        
        output = payload
        for node in topo_order:
            if node in self.agents:
                agent = self.agents[node]
                log.debug("TOPO → %s", node)
                output = agent(memory)
                if tracer:
                    tracer.save(TraceEvent(datetime.now(timezone.utc), node, output, output))
        
        log.info("Topological execution complete")
        return output


class Planner(BaseAgent):
    def __init__(self, generator: Any):
        super().__init__("planner", "Generate agent execution plans.")
        self.llm = generator
        self.portfolio = initialize_agents()

    
    def _prepare_data_for_different_problems(self, data):
        parsed_response = self.llm.generate(
                model_prompt_dir="./",
                prompt_name="parse_component.txt",
                model_args={'response_format': {"type": "json_object"}},
                PREMISE=data['premise'],
                STATEMENT=data['statement']
            )
        
        if not isinstance(parsed_response, dict):
            try:
                spec = json.loads(parsed_response)
            except json.JSONDecodeError as e:
                log.error("Parsing JSON error: %s", parsed_response)
                raise RuntimeError("Parser emitted invalid JSON") from e
        
        # Extract problem list and overall goal from the response
        reasoning_type = parsed_response.get("route", {'reasoning type': 'error', 'reason': ['', '']})
        overall_goal = parsed_response.get("overall_goal", "Solve the reasoning problem")
        
        # instantiate memory 
        memory = Scratchpad()
        
        # Process each problem and store in memory with keyword_id format
        problem_type = reasoning_type.get("reasoning type", "error")
        reason = reasoning_type.get("reason", ["", ""])
        problem_id = 'ques_'+data['id']
        
        # Store problem type for each problem
        if problem_type.lower() == 'epist':
            description = f"Solve epistemic reasoning problem. Determining what is true from mixed or conflicting evidence within the premise. Includes resolution of contradictions between sources, preferring objective measurements (labs, imaging) over opinions, and establishing diagnostic status from an evidence hierarchy."
        elif problem_type.lower() == 'risk':
            description = f"Solve risk reasoning problem. Risk ranking or comparison (highest risk, safer, dangerous), weighing severity against frequency, expected-harm reasoning, and hazards not ruled out by the premise."
        elif problem_type.lower() == 'comp':
            description = f"Solve compositional reasoning problem. Joint constraints over drug–dose–units–schedule–diagnosis–patient factors (age, sex, renal/hepatic function, comorbidities) and co-therapy. Includes dosing bounds, indications, exclusions, and concurrency rules."
        elif problem_type.lower() == 'causal':
            description = f"Solve causal reasoning problem. Statements making causal claims “effect of T on Y” (e.g., cause, lead to, improve, reduce, accelerate;). May include or omit an interventional contrast or comparator to verify."
        
        premise = data['premise']
        statement = data['statement']
        

        problem_record = {
            'problem_id': problem_id,
            'pred_problem_type': problem_type,
            'reason': reason
        }
        problem_record['goal'] = overall_goal
        problem_record['premise'] = premise
        problem_record['statement'] = statement
        problem_record['description'] = description
        problem_record["label"] = data['label']
        problem_record["gt_problem_type"] = data["Reasoning type:"]

        # Write to memory with keyword_id format
        memory.write('GOAL', overall_goal)
        memory.write(f"problem_type_{problem_id}", problem_type)
        memory.write(f'reason_{problem_id}', reason)
        memory.write(f"premise_{problem_id}", premise)
        memory.write(f"statement_{problem_id}", statement)
        memory.write(f"description_{problem_id}", description)
        memory.write(f"problem_record_{problem_id}", problem_record)
            
        return memory, overall_goal, problem_record

    def __call__(self, data) -> List[Plan]:
        memory, goal, problem_record = self._prepare_data_for_different_problems(data)
        memory.write("portfolio", self.portfolio)

        # Prepare problems info for the prompt
        problem_ids = []
        problem_id = problem_record['problem_id']
        problem_ids.append(problem_id)
        
        # Get the goal for this specific problem from memory
        problem_goal_key = f'description_{problem_id}'
        problem_str = memory.read(problem_goal_key) or f"Solve {problem_record['pred_problem_type']} problem."
        
        problem_str = f"QID [{problem_id}]: {problem_str}"
        portfolio_str = ''
        cnt = 1
        for ind, (mod_name, mod_dict) in enumerate(self.portfolio.items()):
            if mod_name != '<PLAN_START>':
                portfolio_str += f"{cnt}. {mod_name}: {mod_dict.get('goal')}\n"
                cnt += 1
        content = self.llm.generate(
                model_prompt_dir="./",
                prompt_name="Planner.txt",
                model_args={'response_format': {"type": "json_object"}},
                GOAL=goal,
                PROBLEM=problem_str,
                PORTFOLIO=portfolio_str,
            )
        
        log.info("[planner] produced plan spec: " + str(content))
        if not isinstance(content, dict):
            try:
                spec = json.loads(content)
            except json.JSONDecodeError as e:
                log.error("Planner JSON error: %s", content)
                raise RuntimeError("Planner emitted invalid JSON") from e
        else:
            spec = content
        # Process the LLM output and add manual connections
        llm_agents = spec.get("agents", [])
        llm_edges = spec.get("edges", [])
        
        # Add <PLAN_START> and ensure all agents are included
        final_agents = list(set(llm_agents + ['<PLAN_START>', '<PLAN_END>']))
        final_edges = [Edge(*pair) for pair in llm_edges]
        
        # Manually connect all problem_ids to <PLAN_START>
        for problem_id in problem_ids:
            if problem_id in llm_agents:
                final_edges.insert(0, Edge('<PLAN_START>', problem_id))
                log.debug(f"Added edge: <PLAN_START> -> {problem_id}")
        
        # Build predecessor mapping from edges
        predecessors_map = defaultdict(list)
        for edge in final_edges:
            predecessors_map[edge.target].append(edge.source)
        
        # Create agent instances
        agents = {}
        
        # Define VirtualAgent that handles problem data initialization
        class VirtualAgent(BaseAgent):
            def __call__(self, memory):
                return
        
        agents_sorted = topological_sort(final_agents, final_edges)
        for agent_name in agents_sorted:
            # Get predecessors for this agent
            agent_predecessors = predecessors_map.get(agent_name, [])
            
            # Calculate problem_to_solve from predecessors
            inherited_problems = []
            for pred_name in agent_predecessors:
                if pred_name in agents:
                    inherited_problems.extend(agents[pred_name].problem_to_solve)
            
            # Remove duplicates while preserving order
            problem_to_solve = list(dict.fromkeys(inherited_problems))
            
            if agent_name in problem_ids:
                # Create virtual agents for problem_ids - they initialize the problem_to_solve
                agents[agent_name] = VirtualAgent(
                    name=agent_name, 
                    goal=f"Virtual agent representing {agent_name}",
                    predecessors=agent_predecessors,
                    problem_to_solve=[agent_name]  # Initialize with the problem ID
                )
            else:
                # Check if agent name has sequence number (e.g., "csp_solver:1")
                base_agent_name = agent_name.split(':')[0] if ':' in agent_name else agent_name
                
                if base_agent_name in self.portfolio:
                    # Get agent class and goal from portfolio
                    agent_class = self.portfolio[base_agent_name]['agent']
                    agent_goal = self.portfolio[base_agent_name]['goal']
                    agent_prompts = self.portfolio[base_agent_name].get('prompt_dict', {})
                    
                    agents[agent_name] = agent_class(
                        name=agent_name,
                        goal=agent_goal,
                        prompt_dict=agent_prompts,
                        generator=self.llm,
                        predecessors=agent_predecessors,
                        problem_to_solve=problem_to_solve
                    )
                else:
                    # This shouldn't happen since control markers are in portfolio
                    log.warning(f"Agent {base_agent_name} not found in portfolio, creating fallback")
                    agents[agent_name] = LLMToolAgent(
                        name=agent_name,
                        goal=f"Fallback agent: {agent_name}",
                        generator=self.llm,
                        predecessors=agent_predecessors,
                        problem_to_solve=problem_to_solve
                    )
        
        return [IterativePlan(agents, final_edges)], memory, problem_ids


# %% Design and Execute Iterative Plans
class IterativePlan(Plan):
    def __init__(self, 
                 agents, 
                 edges,):
        super().__init__(agents, edges)
        self.iteration_start = False
        self.iteration_num = 0

    def _dfs(
        self,
        node: str,
        data: Input,
        memory: Scratchpad,
        tracer: Optional[TracePersister],
    ) -> Output:
        critique_output = memory.read("critique_outputs") or [{},]
        self.max_iterations = memory.read("max_iterations") or 2
        critique_output = critique_output[-1]
        if node == '<ITER_START>':
            log.debug("DFS → %s", node)
            log.debug(f"Start Iteration {self.iteration_num}")
            output = data
            self.iteration_start = True
            self.iter_start_agent = self._graph.get('<ITER_START>', [])[0]

            if tracer:
                tracer.save(TraceEvent(datetime.now(timezone.utc), node, '', ''))
            for child in self._graph.get(node, []):
                output = self._dfs(child, output, memory, tracer)
                
        elif node == '<ITER_END>':
            log.debug("DFS → %s", node)
            log.debug(f"End Iteration {self.iteration_num}")
            output = data
            self.iteration_num += 1
            memory.write("iteration_number", self.iteration_num)

            if tracer:
                tracer.save(TraceEvent(datetime.now(timezone.utc), node, '', ''))
            if not critique_output.get('syntactic validity',True):
                log.info(f"The explanation is not syntactic valid! Continue to the next iteration loop.")
                output = self._dfs('<ITER_START>', output, memory, tracer)
            elif critique_output.get('semantic validity', False):
                log.info(f"The explanation is logically valid! Break the iteration loop.")
                for child in self._graph.get(node, []):
                    output = self._dfs(child, output, memory, tracer)
            else:
                if self.iteration_num >= self.max_iterations:
                    for child in self._graph.get(node, []):
                        output = self._dfs(child, output, memory, tracer)
                else:
                    output = self._dfs('<ITER_START>', output, memory, tracer)
        elif node == '<PLAN_START>' or node == '<PLAN_END>':
            log.debug("DFS → %s", node)
            output = data
            if tracer:
                tracer.save(TraceEvent(datetime.now(timezone.utc), node, '', ''))
            for child in self._graph.get(node, []):
                output = self._dfs(child, output, memory, tracer)
        else:
            agent = self.agents[node]
            log.debug("DFS → %s", node)
            output = agent(memory)
            if tracer:
                tracer.save(TraceEvent(datetime.now(timezone.utc), node, data, output))
            if not critique_output.get('syntactic validity',True) or critique_output.get('semantic validity', False):
                if '<ITER_END>' in self.agents:
                    output = self._dfs('<ITER_END>', output, memory, tracer)
                else:
                    output = self._dfs('<PLAN_END>', output, memory, tracer)
            else:
                for child in self._graph.get(node, []):
                    output = self._dfs(child, output, memory, tracer)
                    
        return output
    