# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from config import hf_config
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.evaluator import Evaluator
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import math
import json
import os
import wandb
import numpy as np
from transformers import get_cosine_schedule_with_warmup

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting training...")


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, train_batches, delimiter_token_id, trainer_ckpt_path=None):
        super().__init__()

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.trainer_ckpt_path = trainer_ckpt_path
        self.train_batches = train_batches
        self.delimiter_token_id = delimiter_token_id
        _, self.hf_conf = hf_config.get_configs(cfg)

    def setup(self, stage):
        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def mask_targets(self, input_ids, target_ids):
        # Find positions with delimiter tokens
        delimiter_positions = (input_ids == self.delimiter_token_id)
        
        # Create a shifted version where positions after delimiter are marked
        # This will include the delimiter itself as True
        first_search_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        first_search_pos[:, 1:] = delimiter_positions.cumsum(dim=1)[:, :-1].bool()
        
        # Create the mask - True for delimiter and positions before it
        mask = ~first_search_pos.cumsum(dim=1).bool()
        
        # Apply the mask to targets, setting masked positions to -100
        return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)


    # def mask_targets(self, input_ids, target_ids):
    #     first_search_pos = (input_ids == self.delimiter_token_id).cumsum(dim=1).bool()
    #     mask = ~first_search_pos.cumsum(dim=1).bool()
    #     # Apply the mask to targets, setting masked positions to -100
    #     return torch.where(mask, torch.tensor(-100, device=target_ids.device), target_ids)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets_no_mask, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets_no_mask)
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        #current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #self.log("learning_rate", current_lr, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        targets = self.mask_targets(idx, targets)
        out, loss = self(idx, targets)
        
        # Compute accuracy only on non-masked tokens
        predictions = torch.argmax(out, dim=-1)
        
        # Create mask for non-masked tokens in the shifted targets
        valid_mask = (targets[:, 1:] != -100)
        
        # Compare predictions with targets only on non-masked positions
        correct = (predictions[:, :-1] == targets[:, 1:]) * valid_mask
        
        # Count total correct predictions and total valid tokens
        total_correct = correct.sum()
        total_tokens = valid_mask.sum()
        
        # Calculate accuracy (avoid division by zero)
        accuracy = total_correct.float() / total_tokens if total_tokens > 0 else torch.tensor(0.0)

        self.log(f"acc", accuracy, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log(f"loss", loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return {f"loss": loss}
    
    def on_validation_epoch_end(self):
        test = self.trainer.datamodule.dataset["test"]
        # print(test["input_ids"])

        save_path = self.cfg.convert_hf.in_path
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)

        self.llm.model.to(self.device)

        evaluator = Evaluator(
            self.cfg,
            test,
            self.preprocessor.tokenizer,
            self.cfg.data.split_str,
            self.global_step,
            self.llm.model,
        )
        
        # Get metrics dictionary from evaluator
        metrics = evaluator.evaluate()
        
        # Log all metrics to wandb/Lightning
        for metric_name, value in metrics.items():
            self.log(
                f"Evaluation/{metric_name}",
                value,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        
        if wandb.run is not None and hasattr(evaluator, 'predictions_after_delimiter'):
            # Create example table
            examples_table = wandb.Table(columns=["Prompt", "Prediction", "Ground Truth", "Exact Match"])
            
            # Select a few random indices to log
            num_examples = min(100, len(evaluator.prompts))
            indices = np.random.choice(len(evaluator.prompts), num_examples, replace=False)
            
            for i in indices:
                prompt = self.preprocessor.tokenizer.decode(evaluator.prompts[i], skip_special_tokens=True)
                pred = evaluator.predictions_after_delimiter[i]
                gt = evaluator.gts[i]
                exact_match = pred == gt
                
                examples_table.add_data(prompt, pred, gt, exact_match)
            
            wandb.log({"evaluation_examples": examples_table}, step=self.global_step)

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(
            self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup_steps
        )
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     n_steps = self.cfg.model.epochs * self.train_batches
    #     warmup_steps = self.train_batches  # 2* epochs worth of steps also viable

    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.lr)
    #     scheduler = {
    #         "scheduler": get_cosine_schedule_with_warmup(
    #             optimizer,
    #             num_warmup_steps=warmup_steps,
    #             num_training_steps=n_steps,
    #         ),
    #         "interval": "step",
    #     }
    #     return [optimizer], [scheduler]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)


@hydra.main(
    config_path="config",
    config_name="config_base",
    version_base=None,
)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers
    tokenizer = get_tokenizer(cfg.tok_data)
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)
    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)
    data.connect(max_seq_length=cfg.model.block_size)
    data.setup()
    train_size = len(data.train_dataloader())
    # print(data.train_dataset["input_ids"])
    trace_start_token_id = tokenizer.encode(cfg.data.split_str, add_special_tokens=False)[0]

    lit_model = LitLLM(model=model, cfg=cfg, train_batches=train_size, preprocessor=preprocessor,
                       delimiter_token_id=trace_start_token_id)

    logger = WandbLogger(
        project="transformer_search_lumi", name=f"{cfg.model.name}", config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="acc",  # what metric to track
        dirpath=f"temp/checkpoints/{cfg.model.name}",  # where to save checkpoints
        filename="{epoch:02d}-{acc:.4f}",  # how to name checkpoints
        save_top_k=2,  # save top 3 models
        mode="max",  # lower val_loss is better
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("total number of params:", total_params)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()