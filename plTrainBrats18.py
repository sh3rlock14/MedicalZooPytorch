import os
import wandb
from pprint import pprint

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import lib.medloaders as medical_loaders
from lib.medloaders.brats2018 import DatasetModule
from lib.medzoo.Unet3Dpl import UNet3Dpl
from lib.losses3D import DiceLoss
from types import SimpleNamespace

from experimentpl import ThesisExperiment

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
    
    # 5.1 LOSS
    # TODO:
    # 1. make the criterion configurable inside the experiment
    # 2. allow for multiple criterions 
    criterion = DiceLoss(classes=config['data_params']['classes'])
    optimizer = "sgd"
    
    experiment = ThesisExperiment(model, criterion, optimizer, config['exp_params'])


    # ----------------------
    # 6 TRAINER
    # ----------------------

    # ckpt formatting
    ckpt_name = config['wandb_params']['ckpt_name']
    
    # 6.1 CALLBACKS
    # TODO: when logging the loss, use the same value specified here!
    callbacks = [
        ModelCheckpoint(
            save_top_k = 2,
            dirpath = os.path.join(config['wandb_params']['save_dir'], "checkpoints"),
            filename = ckpt_name + '-{step}-{loss_rec:.2f}',
            monitor = "dice_loss",
            mode = "min",
            every_n_train_steps= config['trainer_params']['max_steps'] // config['training_params']['n_models_to_save'],
            save_on_train_epoch_end=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]


    

    # NOTE: check wheter or not Validation has been disabled before lunching training
    
    
    if config['training_params']['debug']:
        trainer = Trainer(
            logger= wandb_logger,
            accelerator='gpu',
            devices=1,
            max_steps=10,
            num_sanity_val_steps=2,
        )

    else:
        trainer = Trainer(
            logger= wandb_logger,
            #callbacks = [],
            **config['trainer_params']
        )

    print("So far so good!")
    


    if config['training_params']['resume_train']:
        assert config['training_params']['ckpt_path'] is not None, "You must specify a checkpoint to resume from"
        print(f"====== Resuming Training {config['wandb_params']['model_name']} from {config['training_params']['ckpt_path']} ======")
    else:
        print(f"====== Starting Training {config['wandb_params']['model_name']} from scratch ======")

    
    trainer.fit(
        experiment,
        datamodule = data,
        ckpt_path = config['training_params']['ckpt_path'],
    )

if __name__ == '__main__':


    print(f'Starting a run with:')
    pprint(wandb.config.as_dict())
    main()