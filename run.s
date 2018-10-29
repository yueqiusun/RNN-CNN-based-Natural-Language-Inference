#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=nlp
#SBATCH --mail-type=END
#SBATCH --mail-user=bob.smith@nyu.edu

#SBATCH --output=slurm_%j.out
  
#module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR

for mul in 0 1
do
python main.py --model RNN --num_epochs 10 --hidden_size 200 --kernel_size $ks --mul $mul --learning_rate 0.001 >> result_mul.txt 
done
