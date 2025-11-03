#!/bin/bash
#SBATCH -A rational
#SBATCH -J test_llm_cot
#SBATCH -o ./logger/skip_planner_%j.log
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem 64G

# python _tester.py --backbone_type lunar-gpt-4o --trial_ind 0 --skip_planner --data part2
# python _tester.py --backbone_type lunar-gpt-4o-mini --trial_ind 0 --skip_planner --data part2
# python _tester.py --backbone_type gemini --trial_ind 0 --skip_planner --data part2
# python _tester.py --backbone_type lunar-deepseek-r1 --trial_ind 0 --skip_planner --data part2
# python _tester.py --backbone_type gemini --trial_ind 0 --skip_planner --data part1
# python _tester.py --backbone_type lunar-gpt-4o --trial_ind 0 --skip_planner --data part2 --fixing_trial
# python _tester.py --backbone_type lunar-gpt-4o-mini --trial_ind 0 --skip_planner --data part2 --fixing_trial
python _tester.py --backbone_type gemini --trial_ind 0 --skip_planner --data part2 --fixing_trial
# python _tester.py --backbone_type lunar-deepseek-r1 --trial_ind 0 --skip_planner --data part2 --fixing_trial