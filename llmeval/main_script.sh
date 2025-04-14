#!/bin/bash

set -e

# Script to evaluate a huggingface model or a list of models included in a path
# Takes TASKS, NUM_FEW_SHOT and MODEL_PATH_OR_NAME as arguments.
if [ $# -lt 3 ]; then
    echo "Error: Not enough arguments provided."
    echo "Usage: $0 TASKS NUM_FEWSHOT MODEL_PATH_OR_NAME"
    exit 1
fi

TASKS=$1
NUM_FEWSHOT=$2

# can be a path containing "model.safetensors" files or a model name from huggingface
MODEL_PATH_OR_NAME=$3

# avoid issue "sqlite3.OperationalError: database is locked"
export OUTLINES_CACHE_DIR=/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID/$MODEL_PATH_OR_NAME

mkdir -p $LM_EVAL_OUTPUT_PATH

OUTPUT_PATH=$LM_EVAL_OUTPUT_PATH/$SLURM_ARRAY_JOB_ID/$SLURM_JOB_ID/eval_results/
WANDB_NAME="$SLURM_ARRAY_JOB_ID-$MODEL_PATH"

if [ -d $MODEL_PATH_OR_NAME ]; then
  # we evaluate all models found recursively in MODEL_PATH_OR_NAME
  echo "Evaluating all models found recursively in folder $MODEL_PATH_OR_NAME with $NUM_FEW_SHOT few-shots on the following tasks $TASKS."
  # gets all files "model.safetensors" defined in the current path and launch evaluation for each of them
  mapfile -t SAFETENSOR_FILES < <(find "$MODEL_PATH_OR_NAME" -type f -name "model.safetensors")

  # Check if any files were found
  if [ ${#SAFETENSOR_FILES[@]} -eq 0 ]; then
      echo "No model.safetensors files found in $MODEL_PATH_OR_NAME."
      exit 0
  fi

  echo "Found ${#SAFETENSOR_FILES[@]} model.safetensors files"

  # Loop through all model files found recursively and evaluate all of them
  for FILE_PATH in "${SAFETENSOR_FILES[@]}"; do
    MODEL_PATH=$(dirname "$FILE_PATH")
    echo "Evaluate $MODEL_PATH"
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
        --tasks $TASKS \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --num_fewshot $NUM_FEWSHOT \
        --trust_remote_code \
        --wandb_args project=lm-eval-harness-integration,name=$WANDB_NAME
  done
else
  echo "Evaluate model from huggingface $MODEL_PATH on $TASKS with $NUM_FEW_SHOT fewshots."
  accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
    --tasks $TASKS \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE \
    --num_fewshot $NUM_FEWSHOT \
    --trust_remote_code \
    --wandb_args project=lm-eval-harness-integration,name=$WANDB_NAME
fi

