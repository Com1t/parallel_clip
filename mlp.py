import os
import time
import torch
from torch import nn
import torch.distributed as dist
from transformers import CLIPVisionConfig
from mlp import CLIPMLP, ParallelCLIPMLP


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    # Configuration
    cfg = CLIPVisionConfig()

    cfg.hidden_size = 1024
    cfg.image_size = 336
    cfg.intermediate_size = 4096
    cfg.num_attention_heads = 16
    cfg.num_hidden_layers = 24
    cfg.patch_size = 14
    cfg.projection_dim = 768

    # Example input and configuration
    batch_size = 1
    input_dim = cfg.hidden_size

    input_tensor = torch.zeros([batch_size, 577, input_dim])
    nn.init.xavier_normal_(input_tensor)

    # ensure every rank has the same input tensor
    dist.broadcast(input_tensor, src=0)

    # Instantiate the parallel MLP and local MLP
    local_mlp = CLIPMLP(cfg).to(device)
    local_mlp.weight_init()

    parallel_mlp = ParallelCLIPMLP(cfg).to(device)
    parallel_mlp.weight_init(local_mlp.fc1, local_mlp.fc2)

    # warmup
    num_warmup_runs = 10
    for _ in range(num_warmup_runs):
        with torch.no_grad():
            _ = parallel_mlp(input_tensor)
            _ = local_mlp(input_tensor)
    dist.barrier()

    # Forward pass on GPU
    num_runs = 20
    parallel_time = 0.0
    for _ in range(num_runs):
        dist.barrier()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            parallel_output = parallel_mlp(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()

            parallel_time += end_time - start_time
    parallel_time /= num_runs
    print(f"Rank {rank}: time for parallel MLP: {parallel_time * 1000:.3} ms")

    # Reduce the maximum parallel time to rank 0
    parallel_time_tensor = torch.tensor(parallel_time, device=device)
    dist.reduce(parallel_time_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        max_parallel_time = parallel_time_tensor.item()
        print(f"Max parallel MLP time: {max_parallel_time * 1000:.3} ms")

    local_time = 0.0
    for _ in range(num_runs):
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            local_output = local_mlp(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()

            local_time += end_time - start_time
    local_time /= num_runs
    print(f"Rank {rank}: time for local MLP: {local_time * 1000:.3} ms")

    # Reduce the maximum parallel time to rank 0
    local_time_tensor = torch.tensor(local_time, device=device)
    dist.reduce(local_time_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        max_local_time = local_time_tensor.item()
        print(f"Max local MLP time: {max_local_time * 1000:.3} ms")

    # Verification: Check if the outputs are close
    if rank == 0:
        if torch.allclose(parallel_output, local_output, atol=1e-5):
            print(
                f"Rank {rank}: Verification passed: Parallel and local MLP outputs are close."
            )
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        # print(f"Rank {rank} Local Output:", local_output)
        # print(f"Rank {rank} Parallel Output:", parallel_output)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
