#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:00:00
#SBATCH --job-name tnos_serialx40
#SBATCH --output=tnos_1_40__%j.txt
#SBATCH --mail-type=FAIL


source ~/intel-jax/bin/activate
# Turn off implicit threading in Python, R


cd $SLURM_SUBMIT_DIR

iter=$(($SLURM_ARRAY_TASK_ID*40))
# EXECUTION COMMAND; ampersand off 20 jobs and wait
task=$(expr $iter + 0) && python3 run_stack.py $task &
task=$(expr $iter + 1) && python3 run_stack.py $task &
task=$(expr $iter + 2) && python3 run_stack.py $task &
task=$(expr $iter + 3) && python3 run_stack.py $task &
task=$(expr $iter + 4) && python3 run_stack.py $task &
task=$(expr $iter + 5) && python3 run_stack.py $task &
task=$(expr $iter + 6) && python3 run_stack.py $task &
task=$(expr $iter + 7) && python3 run_stack.py $task &
task=$(expr $iter + 8) && python3 run_stack.py $task &
task=$(expr $iter + 9) && python3 run_stack.py $task &
task=$(expr $iter + 10) && python3 run_stack.py $task &
task=$(expr $iter + 11) && python3 run_stack.py $task &
task=$(expr $iter + 12) && python3 run_stack.py $task &
task=$(expr $iter + 13) && python3 run_stack.py $task &
task=$(expr $iter + 14) && python3 run_stack.py $task &
task=$(expr $iter + 15) && python3 run_stack.py $task &
task=$(expr $iter + 16) && python3 run_stack.py $task &
task=$(expr $iter + 17) && python3 run_stack.py $task &
task=$(expr $iter + 18) && python3 run_stack.py $task &
task=$(expr $iter + 19) && python3 run_stack.py $task &
task=$(expr $iter + 20) && python3 run_stack.py $task &
task=$(expr $iter + 21) && python3 run_stack.py $task &
task=$(expr $iter + 22) && python3 run_stack.py $task &
task=$(expr $iter + 23) && python3 run_stack.py $task &
task=$(expr $iter + 24) && python3 run_stack.py $task &
task=$(expr $iter + 25) && python3 run_stack.py $task &
task=$(expr $iter + 26) && python3 run_stack.py $task &
task=$(expr $iter + 27) && python3 run_stack.py $task &
task=$(expr $iter + 28) && python3 run_stack.py $task &
task=$(expr $iter + 29) && python3 run_stack.py $task &
task=$(expr $iter + 30) && python3 run_stack.py $task &
task=$(expr $iter + 31) && python3 run_stack.py $task &
task=$(expr $iter + 32) && python3 run_stack.py $task &
task=$(expr $iter + 33) && python3 run_stack.py $task &
task=$(expr $iter + 34) && python3 run_stack.py $task &
task=$(expr $iter + 35) && python3 run_stack.py $task &
task=$(expr $iter + 36) && python3 run_stack.py $task &
task=$(expr $iter + 37) && python3 run_stack.py $task &
task=$(expr $iter + 38) && python3 run_stack.py $task &
task=$(expr $iter + 39) && python3 run_stack.py $task &

wait
