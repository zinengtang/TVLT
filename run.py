import os
import json
import copy
import pytorch_lightning as pl

from model.config import ex
from model.modules import Transformer
from model.data.datamodules.multitask_datamodule import MTDataModule
import torch


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = Transformer(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_from_{_config["load_local_path"].split("/")[-1][:-5]}_{_config["model_type"]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["gpus"]
        if isinstance(_config["gpus"], int)
        else len(_config["gpus"])
    )

    total_bs = _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    
    grad_steps = max(_config["batch_size"] // total_bs, 1)
    
    trainer = pl.Trainer(
        gpus=_config["gpus"],
        num_nodes=_config["num_nodes"],
        accelerator="gpu",
        strategy="ddp",
        benchmark=True,
        deterministic=False,
        accumulate_grad_batches=grad_steps,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
