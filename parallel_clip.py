import os
import time
import torch
from torch import nn
import torch.distributed as dist

from PIL import Image
import requests
from transformers import CLIPVisionConfig
from modeling_clip import CLIPVisionModel
from modeling_parallel_clip import ParallelCLIPVisionModel

torch.manual_seed(42)


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

    local_model = CLIPVisionModel(cfg).to(device)
    local_model.vision_model.embeddings.weight_init()
    for layer in local_model.vision_model.encoder.layers:
        layer.self_attn.weight_init()
        layer.mlp.weight_init()
    parallel_model = ParallelCLIPVisionModel(cfg).to(device)
    parallel_model.vision_model.embeddings.class_embedding = (
        local_model.vision_model.embeddings.class_embedding
    )
    parallel_model.vision_model.embeddings.patch_embedding.weight = (
        local_model.vision_model.embeddings.patch_embedding.weight
    )
    parallel_model.vision_model.embeddings.position_embedding.weight = (
        local_model.vision_model.embeddings.position_embedding.weight
    )

    parallel_model.vision_model.embeddings.weight_init()
    for target_layer, raw_layer in zip(
        parallel_model.vision_model.encoder.layers,
        local_model.vision_model.encoder.layers,
    ):
        target_layer.self_attn.weight_init(
            raw_layer.self_attn.q_proj,
            raw_layer.self_attn.k_proj,
            raw_layer.self_attn.v_proj,
            raw_layer.self_attn.o_proj,
        )
        target_layer.mlp.weight_init(raw_layer.mlp.fc1, raw_layer.mlp.fc2)

    batch_size = 1
    inputs = torch.rand(batch_size, 3, 336, 336).to(device)

    # warmup
    num_warmup_runs = 10
    for _ in range(num_warmup_runs):
        with torch.no_grad():
            _ = local_model(inputs)
            _ = parallel_model(inputs)
    dist.barrier()

    print("warmed!")

    # Forward pass on GPU
    num_runs = 20
    parallel_time = 0.0
    for _ in range(num_runs):
        dist.barrier()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            parallel_output = parallel_model(inputs)
            # dist.barrier()
            torch.cuda.synchronize()
            end_time = time.time()

            parallel_time += end_time - start_time
            # if rank == 0:
            #     print(f"Rank {rank}: time for CLIP: {(end_time - start_time) * 1000:.3} ms")
    parallel_time /= num_runs
    print(f"Rank {rank}: time for CLIP: {parallel_time * 1000:.3} ms")

    # Reduce the maximum parallel time to rank 0
    parallel_time_tensor = torch.tensor(parallel_time, device=device)
    dist.reduce(parallel_time_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        max_parallel_time = parallel_time_tensor.item()
        print(f"Max parallel CLIP time: {max_parallel_time * 1000:.3} ms")

    local_time = 0.0
    for _ in range(num_runs):
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            local_output = local_model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()

            local_time += end_time - start_time
    local_time /= num_runs
    print(f"Rank {rank}: time for local CLIP: {local_time * 1000:.3} ms")

    # Reduce the maximum parallel time to rank 0
    local_time_tensor = torch.tensor(local_time, device=device)
    dist.reduce(local_time_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        max_local_time = local_time_tensor.item()
        print(f"Max local CLIP time: {max_local_time * 1000:.3} ms")

    # Verification: Check if the outputs are close
    if rank == 0:
        if torch.allclose(
            parallel_output.last_hidden_state, local_output.last_hidden_state, atol=1e-5
        ):
            print(
                f"Rank {rank}: Verification passed: Parallel and local last_hidden_state outputs are close."
            )
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        # print(f"Rank {rank} Local Output:", local_output)
        # print(f"Rank {rank} Parallel Output:", parallel_output)

        if torch.allclose(
            parallel_output.last_hidden_state, local_output.pooler_output, atol=1e-5
        ):
            print(
                f"Rank {rank}: Verification passed: Parallel and local pooler_output outputs are close."
            )
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        # print(f"Rank {rank} Local Output:", local_output)
        # print(f"Rank {rank} Parallel Output:", parallel_output)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
