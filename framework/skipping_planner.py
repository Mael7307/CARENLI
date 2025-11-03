import json
import yaml
import os
import shutil
from agents.solvers.LLM_solver import LLMSolver
from agents.meta_agents.initializer import initialize_agents
from base.logger import setup_logging
from base.error_collector import setup_error_collection, get_error_collector
from agents.base import Scratchpad
log = setup_logging()

# Setup error collection to capture errors from across the system
error_collector = setup_error_collection()

sample = False



def evaluating_solver(test_data_name, llm, backbone_type = 'api', trial_ind = 0, fixing_trial = False):
    ################### initialization ###################
    # initialize all available agents

    # set result path
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load configuration
    result_path = config.get('cache_dir', {}).get('result_dir')
    path_str = f"results"
    path_str += f'_{test_data_name}'
    path_str += f'_skip_planner'
    path_str += f'_{backbone_type}'
    path_str += f'_{trial_ind}'
    path_str += '.json'
    result_path_agent = os.path.join(result_path, path_str)
    log.info('*'*20+f'Saving to {result_path_agent}'+'*'*20)

    ################### load the test data ###################
    data_dir = config.get('data_dir', {}).get(test_data_name)
    with open(data_dir, "r") as f:
        test_data = json.load(f)

    if isinstance(test_data, dict):
        test_data_list = []
        for id, entry in test_data.items():
            entry['id'] = id
            test_data_list.append(entry)
        test_data = test_data_list
    
    if sample:
        selected_data = []
        reasoning_types = set(entry.get('Reasoning type:', 'unknown') for entry in test_data)
        for reasoning_type in reasoning_types:
            selected_data_category = []
            for entry in test_data:
                if entry.get('Reasoning type:', 'unknown') == reasoning_type:
                    selected_data_category.append(entry)
                    if len(selected_data_category) >= 5:
                        break
            selected_data += selected_data_category
        test_data = selected_data   

    # check if test data has 'id' attribute, if not, create one
    id_check = test_data[0].get('id', None)
    if id_check is None:
        for idx, entry in enumerate(test_data):
            entry['id'] = idx
    
    if fixing_trial:
        ids_to_rerun = [entry['id'] for entry in test_data if entry["Reasoning type:"] not in ['epsitemic', 'risk']]
    else:
        ids_to_rerun = []

    # check if there are already evaluated cases, if so, get their ids
    evaluated_ids = []
    results = []
    if os.path.exists(result_path_agent):
        with open(result_path_agent, "r") as f:
            results = json.load(f)
        results = [res for res in results if res['problem_id'] not in ids_to_rerun]
        results = [res for res in results if res["solution_history"] is not None]
        evaluated_ids = {r["problem_id"] for r in results}
    
    
    ################### testing ###################
    # test all samples
    for idx, entry in enumerate(test_data):
        # check if already evaluated
        if entry['id'] in evaluated_ids:
            log.info(f"Skipping already evaluated case with ID {entry['id']}")
            continue

        # Set session ID for error tracking
        problem_id = entry['id']
        session_id = f"case_{problem_id}"
        error_collector.set_session_id(session_id)
        
        # Clear previous errors for this session
        error_collector.clear_errors()

        if isinstance(entry['statement'], list):
            entry['statement'] = ' '.join(entry['statement'])

        
        if isinstance(entry['premise'], list):
            entry['premise'] = ' '.join(entry['premise'])

        try:
            # parse data for the solver and design the plan

            problem_record = {'problem_id': problem_id, 
                            'premise': entry['premise'], 
                            'statement': entry['statement'], 
                            "label": entry['label'],
                            "gt_problem_type": entry["Reasoning type:"]
                            }

            # initialize memory and put the premise and statement into it
            memory = Scratchpad()
            memory.write(f"premise_ques_{problem_id}", entry['premise'])
            memory.write(f"statement_ques_{problem_id}", entry['statement'])

            # initialize the solver
            portfolio = initialize_agents()
            solver_mapping = {
                "risk":"risk_solver",
                "epsitemic":"epistemic_solver",
                "comp":"compositional_solver",
                "causal":"causal_solver"
            }
            agent_name = solver_mapping.get(entry["Reasoning type:"]).lower()
            agent_class = portfolio[agent_name]['agent']
            agent_goal = portfolio[agent_name]['goal']
            agent_prompts = portfolio[agent_name].get('prompt_dict', {})
            
            solver = agent_class(
                name=agent_name,
                goal=agent_goal,
                prompt_dict=agent_prompts,
                generator=llm,
                predecessors=[f'ques_{problem_id}',],
                problem_to_solve='\n'.join([entry['premise'], entry['statement']])
            )

            # solve 
            output = solver(memory)

            solution_history = memory.read(f"solution_history")
            solution_final = memory.read(f"solution_final")
            problem_record['solution_history'] = solution_history
            problem_record['solution_final'] = solution_final

            gt = entry['label']
            pred = solution_final['pred']
            if gt.lower() == pred.lower():
                is_correct = True
            else:
                is_correct = False
            problem_record['is_correct'] = is_correct
            
            log.info(f'Agentic Solver id ' + entry['id'] + f': predicted: {pred}, groundtruth: {gt}, overall evaluation: {is_correct}')
        except Exception as e:
            import traceback
            # Get the original exception location (for debugging)
            # tb = traceback.extract_tb(e.__traceback__)
            tb = False
            if tb:
                last_frame = tb[-1]
                original_location = f"{last_frame.filename}:{last_frame.lineno}"
                original_function = last_frame.name
                log.error(f'Error on id {entry["id"]} at {original_location} in {original_function}(): {e}', exc_info=True)
            else:
                log.error(f'Error on id ' + entry['id'] + ': '+ f'{e}', exc_info=True)
            if len(problem_record) == 0:
                problem_record = {'problem_id': entry['id'], 
                            'premise': entry['premise'], 
                            'statement': entry['statement'], 
                            "label": entry['label'],
                            "gt_problem_type": entry["Reasoning type:"]
                            }
            else:
                problem_record['problem_id'] = entry['id'].replace('ques_','')
            
            try:
                solution_history = memory.read(f"solution_history_{problem_id}")
                solution_final = memory.read(f"solution_final_{problem_id}")
            except Exception as e:
                log.error(f"Error retrieving solution history or final solution for case {entry['id']}: {e}")
                solution_history = ""
                solution_final = ""
            problem_record['solution_history'] = solution_history
            problem_record['solution_final'] = solution_final

            problem_record['is_correct'] = False
        
        # Collect error information for this test case
        error_summary = error_collector.get_error_summary(session_id)
        problem_record['error_info'] = error_summary['errors']
        
        if error_summary['total_errors'] > 0:
            log.info(f"Case {entry['id']}: Collected {error_summary['total_errors']} errors during processing")
        
        results.append(problem_record)
        # Save after each case to avoid losing progress
        with open(result_path_agent, "w") as f:
            json.dump(results, f, indent=2)

    # Final accuracy
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    log.info(f"Agentic Solver Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")
    return results
