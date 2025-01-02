import os
import time
import torch
from torch import nn
import torch.distributed as dist
from transformers import CLIPVisionConfig
from attention import CLIPAttention, ParallelCLIPAttention


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("gloo")

    device = torch.device("cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)

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
    batch_size = 40
    seq_len = 577

    input_tensor = torch.zeros([batch_size, seq_len, cfg.hidden_size])
    nn.init.xavier_normal_(input_tensor)

    # ensure every rank has the same input tensor
    dist.broadcast(input_tensor, src=0)

    # Instantiate the parallel attn and local attn
    local_attn = CLIPAttention(cfg).to(device)
    local_attn.weight_init()

    parallel_attn = ParallelCLIPAttention(cfg).to(device)
    parallel_attn.weight_init(
        local_attn.q_proj, local_attn.k_proj, local_attn.v_proj, local_attn.o_proj
    )

    # warmup
    with torch.no_grad():
        local_output, _ = local_attn(input_tensor)
        parallel_output, _ = parallel_attn(input_tensor)

    # Forward pass on GPU
    with torch.no_grad():
        if rank == 0:
            torch.cuda.synchronize()
            start_time = time.time()
            local_output, _ = local_attn(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()

            local_time = end_time - start_time
            print(f"time for local attention: {local_time * 1000:.3} ms")

        torch.cuda.synchronize()
        dist.barrier()
        start_time = time.time()
        parallel_output, _ = parallel_attn(input_tensor)
        dist.barrier()
        torch.cuda.synchronize()
        end_time = time.time()

        parallel_time = end_time - start_time
        print(f"Rank {rank}: time for parallel attention: {parallel_time * 1000:.3} ms")

        # Reduce the maximum parallel time to rank 0
        parallel_time_tensor = torch.tensor(parallel_time, device=device)
        dist.reduce(parallel_time_tensor, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            max_parallel_time = parallel_time_tensor.item()
            print(f"Max parallel attention time: {max_parallel_time * 1000:.3} ms")

    # Verification: Check if the outputs are close
    if rank == 0:
        if torch.allclose(parallel_output, local_output, atol=1e-2):
            print(
                f"Rank {rank}: Verification passed: Parallel and local Attention outputs are close."
            )
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        # print(f"Rank {rank} Local Output:", local_output)
        # print(f"Rank {rank} Parallel Output:", parallel_output)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
