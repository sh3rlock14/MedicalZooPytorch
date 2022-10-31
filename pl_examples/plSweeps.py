import argparse
import yaml
import wandb

# ----------------------
# SWEEP CONFIG VARIABLES
# ----------------------

sweepConfigPath = "plConfigs/default_configs.yaml"
defaultConfigPath = ""


# 🐝 Step 2: Load sweep config
parser = argparse.ArgumentParser(description="Sweeper for Medical Image Segmentation Trainer")
parser.add_argument('--sweep_config', '-sc',
                    dest = 'sweep_config',
                    metavar = 'FILE',
                    help = 'path to the sweep config file',
                    default = sweepConfigPath)


args = parser.parse_args()

with open(args.sweep_config, 'r') as file:
    try:
        sweep_config = yaml.safe_load(file)
        print("Sweep config loaded!")
    except yaml.YAMLError as exc:
        print(exc)


projectName = sweep_config['parameters']['sweep_params']['parameters']['project']['value']
nTrials = sweep_config['parameters']['sweep_params']['parameters']['ntrials']['value']


# 🐝 Step 3: Initialize sweep by passing in config
sweep_id = wandb.sweep(
    sweep_config,
    project=projectName,
    )

# 🐝 Step 4: Call to `wandb.agent` to start a sweep
wandb.agent(sweep_id, count = nTrials)