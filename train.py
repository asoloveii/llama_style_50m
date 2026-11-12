import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from torch.utils.data.distributed import DistributedSampler

import hydra
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from functools import partial
from utils import *
from src.llama_style_50m import (
    Transformer,
    ModelConfig,
    alpaca_get_data,
    alpaca_collate_fn,
    fineweb_get_data
)


@hydra.main(version_base=None, config_path="./configs", config_name="pretraining.yaml")
def train(cfg):
    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    rank, local_rank, world_size, device = setup_ddp()
    
    print("Initializing dataset and dataloaders")
    if cfg.dataset_name == "fineweb":
        train_ds, val_ds = fineweb_get_data(cfg.shard_dir, 
                                            cfg.context_size,
                                            cfg.test_size)
        collate_fn = None

        print(f"Total shards found: {len(train_ds.dataset.shard_paths if hasattr(train_ds, 'dataset') else train_ds.shard_paths)}")
        print(f"Total dataset tokens: {train_ds.dataset.total_tokens if hasattr(train_ds, 'dataset') else train_ds.total_tokens:,}")
        print(f"Train dataset size (samples): {len(train_ds):,}")
        print(f"Val dataset size (samples): {len(val_ds):,}")

    elif cfg.dataset_name == "alpaca":
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        train_ds, val_ds = alpaca_get_data(cfg.dataset_path,
                                           tokenizer,
                                           cfg.context_size,
                                           cfg.test_size,
                                           cfg.seed)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        collate_fn = partial(alpaca_collate_fn, pad_token_id=tokenizer.pad_token_id)
    
    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              sampler=sampler,
                              drop_last=True,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            drop_last=True,
                            collate_fn=collate_fn) 
    
    print("Dataloaders are ready!")
    print("Model initialization + optimizer and schedulers")
    
    model = Transformer(ModelConfig()).to(device)

    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # exclude 1d params from weight decay
        if param.ndim == 1 or "norm" in name or "bias" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(optim_groups, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    s1 = optim.lr_scheduler.LinearLR(optimizer, total_iters=cfg.warmup_steps, start_factor=cfg.lr_factor)
    s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_steps-cfg.warmup_steps, eta_min=cfg.eta_min)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[cfg.warmup_steps]) 

    scaler = torch.amp.GradScaler()
    
    epoch, step, run_id = load_ckpt(
        model, optimizer, scheduler, scaler, 
        ckpt_dir=cfg.ckpt_dir,
        device=device,
        rank=rank,
        world_size=world_size
    )
    
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    print("Initialization is completed")
    
    if rank == 0: 
        wandb.init(
            project="llama-style-50m",
            id=run_id,
            resume="allow",
            config=cfg_dict
        )

    for epoch in range(epoch, cfg.epochs):
        sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0

        for i, (xs, ys) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            xs, ys = xs.to(device), ys.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(xs)
                loss = F.cross_entropy(logits.flatten(0, 1), ys.flatten()) 
                loss = loss / cfg.accum_steps
            
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (i+1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

                if rank == 0:                    
                    wandb.log({"train/loss": running_loss, "lr": scheduler.get_last_lr()[0]})
                    running_loss = 0.0

                if (step+1) % cfg.val_every == 0 and rank == 0:
                    with model.no_sync():
                        val_loss = evaluate(model, val_loader, device)
                        wandb.log({"val/loss": val_loss})

                if (step+1) % cfg.ckpt_every == 0 and rank == 0:
                    save_ckpt(model, optimizer, scheduler, scaler, epoch, step, run_id, cfg.ckpt_dir)

    if rank == 0:
        wandb.finish()
    
    destroy_process_group()


@torch.no_grad
def evaluate(model, val_loader, device):
    step_loss = 0.0
    
    model.eval()
    for xs, ys in tqdm(val_loader, desc="Validating"):
        xs, ys = xs.to(device), ys.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(xs)
            loss = F.cross_entropy(logits.flatten(0, 1), ys.flatten())
            step_loss += loss.item()

    step_loss /= len(val_loader)
    return step_loss


if __name__ == "__main__":
    train()
