#!/bin/bash
#SBATCH -A rational
#SBATCH -J test_llm_cot
#SBATCH -o ./logger/gpt4o_%j.log
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem 64G

python _tester.py --backbone_type lunar-gpt-4o --trial_ind 0 --data part2
python _tester.py --backbone_type lunar-gpt-4o --trial_ind 1 --data part2
python _tester.py --backbone_type lunar-gpt-4o --trial_ind 2 --data part2
python _tester.py --backbone_type lunar-gpt-4o --trial_ind 3 --data part2
python _tester.py --backbone_type lunar-gpt-4o --trial_ind 4 --data part2