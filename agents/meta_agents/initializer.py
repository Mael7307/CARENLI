

# from agents.solvers.solver_utils.isabelle import IsabelleCritique
from agents.solvers.LLM_solver import LLMSolver


from agents.base import LLMToolAgent, Portfolio, BaseAgent, SpecialControlMarker
from base.logger import setup_logging
log = setup_logging()

import os
import yaml

import os
import subprocess


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def register_agent(portfolio: Portfolio, agent: BaseAgent, name: str, goal: str, prompt_dict: dict = {}) -> None:
    if name in portfolio.keys():
        raise ValueError(f"Agent name collision: {name}")
    portfolio[name] = {
        'goal': goal,
        'agent': agent,
        'prompt_dict':prompt_dict
        }
    log.info("Registered agent '%s'", name)



def initialize_agents():
    # Seed portfolio
    portfolio: Portfolio = {}

    register_agent(portfolio, 
        SpecialControlMarker,
        name = "<PLAN_START>", 
        goal = "Special control marker, indicating the start of the whole working plan."
        )
    register_agent(portfolio, 
        SpecialControlMarker,
        name = "<PLAN_END>", 
        goal = "Special control marker, indicating the end of the whole working plan."
        )

    register_agent(portfolio, 
        LLMSolver,
        name = "epistemic_solver", 
        goal = "Determining what is true from mixed or conflicting evidence within the premise. Includes resolution of contradictions between sources, preferring objective measurements (labs, imaging) over opinions, and establishing diagnostic status from an evidence hierarchy.",
        prompt_dict={
            'get_solution': 'Epistemic_solver.txt',
            'get_refinement': 'Epistemic_refiner.txt',
            'get_verification': 'Epistemic_verifier_merged.txt',
        }
        )
    register_agent(portfolio, 
        LLMSolver,
        name = "risk_solver", 
        goal = "Risk ranking or comparison (highest risk, safer, dangerous), weighing severity against frequency, expected-harm reasoning, and hazards not ruled out by the premise.",
        prompt_dict={
            'get_solution': 'Risk_solver.txt',
            'get_refinement': 'Risk_refiner.txt',
            'get_verification': 'Risk_verifier_merged.txt'
        }
        )
    register_agent(portfolio, 
        LLMSolver,
        name = "compositional_solver",
        goal = "Joint constraints over drug–dose–units–schedule–diagnosis–patient factors (age, sex, renal/hepatic function, comorbidities) and co-therapy. Includes dosing bounds, indications, exclusions, and concurrency rules.",
        prompt_dict={
            'get_solution': 'Composition_solver.txt',
            'get_refinement': 'Composition_refiner.txt',
            'get_verification': 'Composition_verifier_merged.txt'
        }
        ) 
    register_agent(portfolio, 
        LLMSolver,
        name = "causal_solver",
        goal = "Statements making causal claims “effect of T on Y” (e.g., cause, lead to, improve, reduce, accelerate;). May include or omit an interventional contrast or comparator to verify.",
        prompt_dict={
            'get_solution': 'Causal_solver.txt',
            'get_refinement': 'Causal_refiner.txt',
            'get_verification': 'Causal_verifier_merged.txt'
        }
        ) 
    return portfolio