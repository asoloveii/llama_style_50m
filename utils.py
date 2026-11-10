import os
import wandb
import torch
from wandb.sdk.lib.runid import generate_id
from torch.distributed import init_process_group


def setup_ddp():
    init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, local_rank, world_size, device


def save_ckpt(model,
              optimizer, 
              scheduler,
              scaler,
              epoch,
              step,
              run_id,
              ckpt_dir):
    torch.save({
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "run_id": run_id
    }, f"./{ckpt_dir}/ckpt_{step:06d}")


def load_ckpt(model,
              optimizer,
              scheduler,
              scaler,
              ckpt_dir,
              device,
              rank,
              world_size):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_")])
    
    if len(ckpt_files) == 0:
        print("No checkpoints found. Starting a fresh training run...")
        epoch = 0
        step = 0
        
        run_id = generate_id() if rank == 0 else None
    else:
        latest_path = os.path.join(ckpt_dir, ckpt_files[-1])
        print(f"Resuming training from checkpoint: {latest_path}")
        
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        
        epoch = state["epoch"]
        step = state["step"]
        run_id = state["run_id"]
        
    return epoch, step, run_id
