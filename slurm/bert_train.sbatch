#!/usr/bin/env bash
#SBATCH --output=jobs/bert_cls/%J_slurm.out
#SBATCH --error=jobs/bert_cls/%J_slurm.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen"
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
BERT_DATA_DIR=/Midgard/home/martinig/thesis-src/data/jigsaw/bert_classifier_training
BERT_MODEL_DIR=/Midgard/home/martinig/transformer-drg-style-transfer/bert_models/toxic

mkdir -p $BERT_MODEL_DIR

python run_classifier.py \
--data_dir=$BERT_DATA_DIR \
--bert_model=bert-base-uncased \
--task_name=jigsaw \
--output_dir=$BERT_MODEL_DIR \
--max_seq_length=70 \
--do_train \
--do_lower_case \
--train_batch_size=32 \
--num_train_epochs=1