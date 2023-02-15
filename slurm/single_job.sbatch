#!/usr/bin/env bash
#SBATCH --output=jobs/delete_and_generate/%J_slurm.out
#SBATCH --error=jobs/delete_and_generate/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="rivendell"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH -t 0-4:00  # time limit: (D-HH:MM) 


# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate transformer-drg-style-transfer

# Train
DG_DATA_PATH=/Midgard/home/martinig/transformer-drg-style-transfer/data/yelp/processed_files_with_bert_with_best_head
DG_MODEL_PATH=/Midgard/home/martinig/transformer-drg-style-transfer/models
DG_DEV_DATA="${DG_DATA_PATH}/sentiment.dev"
DG_TRAIN_DATA="${DG_DATA_PATH}/sentiment.train"
DG_MODEL_OUT="${DG_MODEL_PATH}/delete_and_generate"

mkdir -p $DG_MODEL_OUT

python openai_gpt_delete_and_generate.py \
--model_name ~/.pytorch_pretrained_bert \
--do_train \
--do_eval \
--train_dataset $DG_TRAIN_DATA \
--eval_dataset $DG_DEV_DATA \
--train_batch_size 32 \
--eval_batch_size 32 \
--max_seq_length 85 \
--num_train_epochs 2 \
--output_dir $DG_MODEL_OUT 