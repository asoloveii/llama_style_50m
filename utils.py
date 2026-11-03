import torch
from torch.distributed import init_process_group


def setup_ddp():
    init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"]
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
              ckpt_dir):
    latest_path = os.path.joiin(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])

    state = torch.load(latest_path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    
    return state["epoch"], state["step"], state["run_id"]

