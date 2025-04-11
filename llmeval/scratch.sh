#!/usr/bin/env bash
python launch_eval.py \
--model_file ../model_missing_nemotron.txt \
--cluster booster \
--partition booster \
--account laionize \
--hf_home /p/home/jusers/salinas2/juwels/salinas2/oellm_evals/hf/ \
--venv_path /p/home/jusers/salinas2/juwels/salinas2/oellm_evals/venv/  \
--eval_output_path /p/home/jusers/salinas2/juwels/salinas2/oellm_evals/evals