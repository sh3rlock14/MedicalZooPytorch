import os
import wandb
from pprint import pprint

from pytorch_lightning.loggers import WandbLogger


import lib.medloaders as medical_loaders
from lib.medloaders.brats2018 import DatasetModule
from lib.medzoo.Unet3Dpl import UNet3Dpl
from types import SimpleNamespace

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb.init()


def main():
    
    # ----------------------
    # 1 SWEEP CONFIG VARIABLES
    # ----------------------
    config = wandb.config.as_dict()

    # ----------------------
    # 2 DATA PIPELINE
    # ----------------------

    data = DatasetModule(**config['data_params'])
    data.setup() # FOR DEBUG PURPOSES: ONCE DONE, DEL THIS LINE

    # ----------------------
    # 3 WANDB LOGGER
    # ----------------------
    wandb_logger = WandbLogger(
        project = config['sweep_params']['project'],
        save_dir = config['wandb_params']['save_dir'],
        log_model = "all" ) # log while training
    

    # ----------------------
    # 4 MODEL
    # ----------------------
    model = UNet3Dpl(**config['model_params'])


    # ----------------------
    # 5 LIGHTNING EXPERIMENT
    # ----------------------
    experiment = ""


    print("So far so good!")
    #print("Hello World!")

if __name__ == '__main__':


    print(f'Starting a run with:')
    pprint(wandb.config.as_dict())
    main()