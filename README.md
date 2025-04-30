# LLMSlurmEval

This collect scripts to launch evaluations on a list of datasets and checkpoints with a slurm job array.

## Setup

### Node setup

You should first make sure you have a valid python environments and datasets downloaded as internet is not available
on Slurm nodes.

You can do the following:
```
ssh leonardo
git clone XXX
VENV_DIR=~/llmeval/venv/  # where you want to setup a virtual env
HF_HOME=~/llmeval/hf/  # where you want to download HF datasets
bash llm_eval/setup_node.sh $VENV_DIR $HF_HOME
```

which
1) creates an environment 
2) install dependencies
3) download datasets (as you wont have internet access in worker nodes in Slurm clusters)
 

### Launching evaluations

You can now launch the experiments, first install slurmpilot and then call:
```bash
pip install "slurmpilot[extra] @ git+https://github.com/geoalgo/slurmpilot.git"
python launch_eval.py \
--model EleutherAI/pythia-160m,revision=step100000 \
--cluster XXX \
--partition XXX \
--account XXX \
--hf_home $HF_HOME \
--venv_path $VENV_DIR  \
--eval_output_path ~/evals/ \
```

You can alternatively pass `--model_file models.txt` instead of `--model`. The provided file should contain one path
per line.

which will launch all evaluations.

Results will be logged in wandb but you will have to sync them as nodes are cut from internet.
Once results are in WANDB, you can do the following to get the table of all results (you should update the list of 
jobids manually).

## Analysing results

TODO describe [evaluation-analysis.ipynb](llmeval%2Fevaluation-analysis.ipynb)
and how to pull data.