
from ProcessedDataset.WellLogDataLoader import WellLogDataLoader
from model.Diffusion import Diffusion
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model.CVAE import CVAE
from model.CGAN import CGAN

def main(window_size = 500):

    test_dataset = WellLogDataLoader('test', window_size=window_size, step=window_size)
    train_dataset = WellLogDataLoader('train', window_size=window_size, step=window_size)
    val_dataset = WellLogDataLoader('valid', window_size=window_size, step=window_size)
    params = {
        'beta_schedule': 'quad',
        'beta_end': 0.02,
        'diff_steps': 500,
        'target_mask_type': 'interval',
        'target_dim' : 9,
        'episonparams':{
        "layers": 4,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        }
    }

    model_save_path = f'./saved_retrained_model/interval_mask_window_size_{window_size}/'

    logger = TensorBoardLogger("logs", name=model_save_path)

    checkpoint_callback = ModelCheckpoint(
        save_top_k= 1,
        monitor="val_loss",
        mode="min",
        dirpath=model_save_path,
        filename="best_model",
    )

    model = Diffusion(params, train_dataset, val_dataset)
    trainer = Trainer(accelerator="cuda", 
                        #  devices=[1], 
                         max_epochs=1000, 
                         logger = logger, callbacks=[checkpoint_callback],
                        #  strategy = DDPStrategy(find_unused_parameters=False),
                        #  precision=16,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
    