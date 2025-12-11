# 1.2 GPU Acceleration & Optimization (14%)

## Parallelism Strategies

### Tensor Parallel (TP)
- Splits model layers across GPUs
- Each GPU holds a slice of weight matrices
- Requires high-bandwidth interconnect (NVLink)
- Best for large layers that don't fit on single GPU

### Pipeline Parallel (PP)
- Splits model by layers across GPUs
- GPU 1 holds layers 1-10, GPU 2 holds layers 11-20, etc.
- Introduces pipeline bubbles
- Good for very deep models

### Data Parallel (DP)
- Each GPU holds full model copy
- Batch split across GPUs
- Gradients synchronized after backward pass
- Scales with batch size

### Sequence Parallelism (SP)
- Splits sequence dimension across GPUs
- Reduces activation memory
- Often combined with Tensor Parallel

---

## NeMo 2.0 Configuration

- Python-based config (replaces YAML)
- `NeMo-Run` for job orchestration
- `AutoModel` for simplified model loading
- Megatron-Core integration

---

## Batch Configuration

- `micro_batch_size` - samples per GPU per step
- `global_batch_size` - total samples per step
- Gradient accumulation steps
- Memory vs throughput tradeoffs

---

## Nsight Metrics

### SM Occupancy
- Percentage of active warps vs maximum
- Higher is generally better
- Target: >50%

### Kernel Launch Latency
- Time to launch GPU kernels
- Minimize with CUDA graphs
- Target: <10μs

### Memory Bandwidth
- GB/s utilization
- H100: ~3.35 TB/s HBM3
- A100: ~2 TB/s HBM2e

### Compute vs Memory Bound
- Compute bound: increase parallelism
- Memory bound: optimize data movement

---

## Memory Calculations

- Model memory = Parameters × Bytes per parameter
- Optimizer states = 2-3× model size (Adam)
- Activations = Batch × Sequence × Hidden × Layers
- KV Cache = 2 × Layers × Heads × Head_dim × Sequence × Batch

---

## Hardware Knowledge

### NVLink vs PCIe
- **NVLink**: 900 GB/s (H100), GPU-to-GPU
- **PCIe 5.0**: 64 GB/s, CPU-GPU connection
- Use NVLink for Tensor Parallel

### NCCL
- NVIDIA Collective Communications Library
- AllReduce, AllGather, Broadcast
- Optimized for multi-GPU/multi-node

### Tensor Cores
- Specialized matrix multiply units
- FP16, BF16, FP8, INT8 acceleration
- 4x4 matrix operations

### DGX Architecture
- **DGX H100**: 8x H100 GPUs, NVSwitch
- **DGX SuperPOD**: Multiple DGX systems
- InfiniBand interconnect for multi-node

---

## Memory Optimization Techniques

### Gradient Checkpointing (Activation Checkpointing)

Trade compute for memory by recomputing activations during backward pass.

```python
# PyTorch gradient checkpointing
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # Checkpoint attention - activations recomputed during backward
        x = checkpoint(self.attention, x, use_reentrant=False)
        x = checkpoint(self.feedforward, x, use_reentrant=False)
        return x
```

**Memory Savings:**
```
Without checkpointing: O(layers × batch × seq × hidden)
With checkpointing:    O(sqrt(layers) × batch × seq × hidden)

Example 70B model:
- Without: ~200 GB activations
- With:    ~40 GB activations
```

**Trade-offs:**

| Aspect | Without Checkpoint | With Checkpoint |
|--------|-------------------|-----------------|
| Memory | High | ~30-50% reduction |
| Compute | 1x | ~1.3x (recompute) |
| Speed | Faster | ~20-30% slower |

### Selective Checkpointing

Checkpoint only memory-intensive layers:

```python
# Checkpoint every nth layer
for i, layer in enumerate(self.layers):
    if i % checkpoint_every == 0:
        x = checkpoint(layer, x)
    else:
        x = layer(x)
```

### NeMo Activation Checkpointing

```python
from nemo.collections.llm import GPTConfig

config = GPTConfig(
    # Checkpoint options
    activations_checkpoint_method="uniform",  # or "block"
    activations_checkpoint_num_layers=1,      # checkpoint every N layers
    activations_checkpoint_granularity="selective",  # or "full"
)
```

---

## FlashAttention

Memory-efficient attention with IO-awareness.

### Standard Attention Memory Issue

```
Standard attention:
Q, K, V: (batch, heads, seq, head_dim)
Attention matrix: (batch, heads, seq, seq) ← O(seq²) memory!

For seq=8192, heads=32, batch=4:
Attention matrix = 4 × 32 × 8192 × 8192 × 2 bytes = 17 GB per layer!
```

### FlashAttention Solution

```
FlashAttention:
- Tiles Q, K, V into blocks that fit in SRAM
- Computes attention incrementally
- Never materializes full attention matrix
- Memory: O(seq) instead of O(seq²)
```

### Key Techniques

1. **Tiling**: Compute attention in blocks
2. **Recomputation**: Recompute attention in backward pass
3. **Kernel Fusion**: Fuse softmax, dropout, matmul

```python
# Using FlashAttention
from flash_attn import flash_attn_func

# Standard usage
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=1.0 / math.sqrt(head_dim),
    causal=True  # For autoregressive models
)
```

### FlashAttention-2 Improvements

| Feature | FA-1 | FA-2 |
|---------|------|------|
| Parallelism | Batch, heads | + Sequence |
| Work partitioning | Fixed | Adaptive |
| Speedup vs standard | 2-4x | 2-3x over FA-1 |

### TensorRT-LLM FlashAttention

```bash
# Enable during build
trtllm-build \
    --model llama-70b \
    --use_flash_attention \
    --context_fmha enable
```

---

## NCCL Optimization

### Key Collective Operations

```
AllReduce: Sum gradients across all GPUs
           GPU0: [1,2] + GPU1: [3,4] → All GPUs: [4,6]

AllGather: Gather data from all GPUs
           GPU0: [A], GPU1: [B] → All GPUs: [A,B]

ReduceScatter: Reduce then scatter
               Sum then distribute portions

Broadcast: One GPU to all GPUs
```

### NCCL Environment Variables

```bash
# Performance tuning
export NCCL_DEBUG=INFO                    # Debug output
export NCCL_IB_DISABLE=0                  # Enable InfiniBand
export NCCL_NET_GDR_LEVEL=2               # GPUDirect RDMA level
export NCCL_SOCKET_IFNAME=eth0            # Network interface
export NCCL_BUFFSIZE=8388608              # Buffer size (8MB)
export NCCL_NTHREADS=512                  # Thread count

# For multi-node
export NCCL_IB_HCA=mlx5                   # InfiniBand adapter
export NCCL_IB_GID_INDEX=3                # GID index for RoCE
```

### NCCL Optimization Tips

| Scenario | Optimization |
|----------|--------------|
| Slow AllReduce | Enable NVLink, use NCCL_ALGO=Ring |
| Multi-node slow | Check InfiniBand, enable GPUDirect |
| OOM in NCCL | Reduce NCCL_BUFFSIZE |
| High latency | Use NCCL_MIN_NCHANNELS |

---

## Parallelism Selection Guide

### When to Use Each Strategy

```
Model fits on 1 GPU:
  → Data Parallel (DP) only

Model doesn't fit on 1 GPU:
  → Add Tensor Parallel (TP)
  → TP degree = number of GPUs per node with NVLink

Very large model (100B+):
  → TP within node + Pipeline Parallel across nodes
  → PP avoids cross-node communication during forward

Long sequences:
  → Add Sequence Parallel (SP) with TP
  → SP parallelizes LayerNorm and Dropout
```

### Parallelism Configuration Example

```python
# 70B model on 32 GPUs (4 nodes × 8 GPUs)
parallelism = {
    "tensor_parallel": 8,      # Within each node
    "pipeline_parallel": 4,    # Across nodes
    "data_parallel": 1,        # 32 / (8 × 4) = 1
    "sequence_parallel": True, # Combined with TP
}

# Effective: 8 × 4 × 1 = 32 GPUs
```

### Parallelism Performance Trade-offs

| Strategy | Communication | Memory | Best For |
|----------|--------------|--------|----------|
| DP | All-Reduce gradients | High (full model per GPU) | Small models, large batches |
| TP | All-Reduce activations | Medium | Large layers, NVLink |
| PP | Point-to-point activations | Low | Deep models, multi-node |
| SP | None (with TP) | Lower activations | Long sequences |

---

## GPU Memory Breakdown

### Training Memory Components

```
Total GPU Memory = Model + Optimizer + Gradients + Activations + Temp

Model Parameters:
  - FP32: 4 bytes per param
  - BF16: 2 bytes per param

Optimizer States (Adam):
  - Momentum: 4 bytes per param
  - Variance: 4 bytes per param
  - Master weights: 4 bytes (mixed precision)

Gradients:
  - Same precision as training (2-4 bytes)

Activations:
  - Proportional to batch × seq × hidden × layers
```

### Memory Calculation Example

```python
def estimate_training_memory(params_b, batch_size, seq_len, hidden, layers):
    """Estimate GPU memory for training (in GB)"""

    # Model (BF16)
    model_mem = params_b * 2

    # Optimizer states (FP32)
    optimizer_mem = params_b * 12  # momentum + variance + master weights

    # Gradients (BF16)
    gradient_mem = params_b * 2

    # Activations (rough estimate)
    activation_mem = batch_size * seq_len * hidden * layers * 2 / 1e9

    total = model_mem + optimizer_mem + gradient_mem + activation_mem
    return total

# Example: 7B model
memory_gb = estimate_training_memory(
    params_b=7,
    batch_size=4,
    seq_len=4096,
    hidden=4096,
    layers=32
)
# ~120 GB without checkpointing
```

### Inference Memory (Much Lower)

```
Inference Memory = Model + KV Cache

KV Cache per token:
  = 2 × layers × heads × head_dim × bytes
  = 2 × 32 × 32 × 128 × 2 (BF16)
  = 0.5 MB per token for 7B model

For 4K context:
  KV Cache = 4096 × 0.5 MB = 2 GB
```

---

## Performance Benchmarks

### GPU Comparison for LLM Training

| GPU | Memory | Bandwidth | FP16 TFLOPS | BF16 TFLOPS |
|-----|--------|-----------|-------------|-------------|
| A100 40GB | 40 GB | 1.6 TB/s | 312 | 312 |
| A100 80GB | 80 GB | 2.0 TB/s | 312 | 312 |
| H100 SXM | 80 GB | 3.35 TB/s | 990 | 990 |
| H100 PCIe | 80 GB | 2.0 TB/s | 756 | 756 |

### Interconnect Comparison

| Interconnect | Bandwidth | Latency | Use Case |
|--------------|-----------|---------|----------|
| NVLink 4.0 (H100) | 900 GB/s | <1 μs | Intra-node TP |
| NVLink 3.0 (A100) | 600 GB/s | <1 μs | Intra-node TP |
| PCIe 5.0 | 64 GB/s | ~1 μs | CPU-GPU |
| InfiniBand HDR | 200 Gb/s | ~1 μs | Multi-node |
| InfiniBand NDR | 400 Gb/s | <1 μs | Multi-node |

---

## Troubleshooting GPU Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| OOM during training | CUDA OOM error | Reduce batch, enable checkpointing |
| Slow training | Low GPU utilization | Check data loading, increase batch |
| NCCL timeout | Collective operation hung | Check network, increase timeout |
| Low Tensor Core usage | SM active but slow | Ensure BF16/FP16, check alignment |
| Memory fragmentation | OOM with free memory | Use memory pools, restart |
| NVLink not used | Slow TP communication | Check topology, set CUDA_VISIBLE_DEVICES |
