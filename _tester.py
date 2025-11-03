import os

import warnings
warnings.filterwarnings("ignore")

from agents.generation.api import AzureGenerator, GeminiGenerator
import yaml
import argparse

from framework.carnap import evaluating_agent
from framework.skipping_planner import evaluating_solver
from base.logger import setup_logging
log = setup_logging()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_type", type=str, default="gemini") # change this 
    parser.add_argument("--data", type=str, default="part2") # change this 
    parser.add_argument("--trial_ind", type=int, default=4)
    parser.add_argument("--skip_planner", action='store_true')
    parser.add_argument("--fixing_trial", action='store_true')

    args = parser.parse_args()

    # load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if 'lunar' in args.backbone_type:
        api_key = config.get('api_config', {}).get(args.backbone_type, {}).get('api_key')
        model_name = config.get('api_config', {}).get(args.backbone_type, {}).get('model_name')
        model_version = config.get('api_config', {}).get(args.backbone_type, {}).get('version')
        endpoint = config.get('api_config', {}).get(args.backbone_type, {}).get('azure_endpoint')
        llm = AzureGenerator(model_name = model_name, api_key = api_key, model_version = model_version, endpoint = endpoint)
    elif args.backbone_type == 'gemini':
        api_key = config.get('api_config', {}).get('gemini', {}).get('api_key')
        llm = GeminiGenerator(model_name = 'gemini', api_key = api_key)

    log.info('='*50)
    log.info(f"Using {args.backbone_type} model for generation.")
    log_str = f"Testing Carnap."
    log.info(log_str)
    log.info('='*50)
    
    if args.skip_planner:
        log.info("Skipping planner, evaluating solver directly.")
        evaluating_solver(args.data, llm, backbone_type = args.backbone_type, trial_ind = args.trial_ind, fixing_trial=args.fixing_trial)
    else:
        log.info("Evaluation Carnap with planner.")
        evaluating_agent(args.data, llm, backbone_type = args.backbone_type, trial_ind = args.trial_ind)
