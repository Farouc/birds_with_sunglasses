#!/bin/bash
#SBATCH --job-name=farouk_trait
#SBATCH --output=logs/my_model_%j.out
#SBATCH --error=logs/my_model_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=main              # <-- partition valide
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread

echo "===== [INFO] Job started at $(date) on $(hostname) ====="


cd ~/Automathon_Sujet
echo "===== [INFO] Running script: my_model_plus_rf.py ====="

#python my_model_plus_rf.py
# python /automathon/users/mig_user3/Automathon_Sujet/classifier_3rd_method.py
python traitement.py

echo "===== [INFO] Job finished at $(date) ====="
