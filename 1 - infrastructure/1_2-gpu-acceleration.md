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

## CUDA Fundamentals

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform.

### Programming Model

```
CPU (Host)                    GPU (Device)
    │                             │
    ├─ Allocate GPU memory        │
    ├─ Copy data to GPU ─────────→│
    ├─ Launch kernel ────────────→│ Execute in parallel
    ├─ Copy results back ←────────│
    └─ Free GPU memory            │
```

### Thread Hierarchy

```
Grid (all threads)
  └── Block (group of threads)
        └── Thread (single execution unit)

Grid: 2D/3D array of blocks
Block: Up to 1024 threads
Warp: 32 threads executing in lockstep
```

### Memory Hierarchy

| Memory Type | Scope | Speed | Size |
|-------------|-------|-------|------|
| Registers | Thread | Fastest | ~256KB/SM |
| Shared Memory | Block | Fast | 48-164KB/SM |
| L1 Cache | SM | Fast | Combined w/ shared |
| L2 Cache | Device | Medium | 40-50MB |
| Global Memory (HBM) | Device | Slow | 40-80GB |

### Basic CUDA Kernel

```cpp
// Kernel definition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch kernel
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

### PyTorch CUDA Operations

```python
import torch

# Check CUDA availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Move tensors to GPU
device = torch.device("cuda:0")
x = torch.randn(1000, 1000, device=device)

# Or move existing tensor
y = torch.randn(1000, 1000)
y = y.to(device)
y = y.cuda()  # Alternative

# Synchronize (wait for GPU operations)
torch.cuda.synchronize()

# Memory management
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
torch.cuda.empty_cache()
```

### CUDA Streams

Enable concurrent execution of operations.

```python
# Create streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Run operations concurrently
with torch.cuda.stream(stream1):
    output1 = model1(input1)

with torch.cuda.stream(stream2):
    output2 = model2(input2)

# Synchronize streams
torch.cuda.synchronize()
```

### CUDA Graphs

Capture and replay sequences of GPU operations.

```python
# Capture graph
g = torch.cuda.CUDAGraph()

# Warmup
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)

# Capture
with torch.cuda.graph(g):
    output = model(static_input)

# Replay (faster than re-launching kernels)
g.replay()
```

**Benefits:**
- Reduces CPU overhead
- Eliminates kernel launch latency
- Best for repeated operations with same shapes

---

## cuDNN Fundamentals

cuDNN (CUDA Deep Neural Network library) provides optimized primitives for deep learning.

### Key Operations

| Operation | Function | Use Case |
|-----------|----------|----------|
| Convolution | `cudnnConvolutionForward` | CNNs, vision models |
| RNN/LSTM | `cudnnRNNForward` | Sequence models |
| Attention | `cudnnMultiHeadAttnForward` | Transformers |
| Normalization | `cudnnBatchNormalization` | BatchNorm, LayerNorm |
| Activation | `cudnnActivationForward` | ReLU, GELU, etc. |
| Pooling | `cudnnPoolingForward` | Max/Avg pooling |

### cuDNN Convolution Algorithms

```python
# PyTorch cuDNN benchmark mode
torch.backends.cudnn.benchmark = True  # Auto-tune algorithms

# Deterministic mode (reproducibility)
torch.backends.cudnn.deterministic = True
```

**Algorithm Selection:**
- `IMPLICIT_GEMM`: General, works with any size
- `IMPLICIT_PRECOMP_GEMM`: Faster for some sizes
- `GEMM`: Direct matrix multiplication
- `FFT`: Fast Fourier Transform based
- `WINOGRAD`: Efficient for small filters

### cuDNN + Transformers

```python
# Enable cuDNN for attention (PyTorch 2.0+)
torch.backends.cuda.enable_flash_sdp(True)   # Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Check which backend is used
from torch.backends.cuda import (
    flash_sdp_enabled,
    mem_efficient_sdp_enabled
)
```

### cuDNN Best Practices

1. **Enable benchmark mode** for static input sizes
2. **Use FP16/BF16** for Tensor Core acceleration
3. **Align dimensions** to multiples of 8 for Tensor Cores
4. **Warm up** before timing measurements

---

## Memory Optimization Techniques

### Gradient Checkpointing

Trade compute for memory by recomputing activations.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return x
```

**Memory savings:** Up to 5x reduction
**Compute overhead:** ~30% increase

### Mixed Precision Training

Use FP16/BF16 for faster computation with FP32 master weights.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Memory-Efficient Attention

```python
# PyTorch 2.0+ scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention

# Automatically selects best implementation:
# - Flash Attention (memory efficient)
# - Memory-efficient attention
# - Math attention (fallback)
output = scaled_dot_product_attention(query, key, value)
```

---

## Profiling Tools

### NVIDIA Nsight Systems

System-wide performance analysis.

```bash
# Profile Python script
nsys profile -o report python train.py

# View report
nsys stats report.nsys-rep

# Key metrics:
# - CUDA API calls
# - Kernel execution time
# - Memory transfers
# - CPU/GPU timeline
```

### NVIDIA Nsight Compute

Detailed kernel-level analysis.

```bash
# Profile specific kernels
ncu --set full -o profile python train.py

# Key metrics:
# - SM utilization
# - Memory throughput
# - Occupancy
# - Warp stalls
```

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total"))

# Export for visualization
prof.export_chrome_trace("trace.json")
```

### Key Profiling Metrics

| Metric | Good Value | Issue if Low |
|--------|------------|--------------|
| SM Occupancy | >50% | Not enough parallelism |
| Memory Bandwidth | >70% | Memory bound |
| Compute Throughput | >70% | Compute bound |
| Tensor Core Usage | >50% | Missing Tensor Core ops |

---

## Performance Optimization Checklist

1. **Use Tensor Cores:** Align to multiples of 8, use FP16/BF16
2. **Enable cuDNN benchmark:** `torch.backends.cudnn.benchmark = True`
3. **Use CUDA Graphs:** For repeated operations
4. **Mixed precision:** AMP for training
5. **Gradient checkpointing:** For memory-limited scenarios
6. **Optimize batch size:** Maximize GPU utilization
7. **Profile first:** Identify bottlenecks before optimizing
8. **Fuse operations:** Reduce kernel launches

---

## Multi-Node Scaling

### Multi-Node Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Multi-Node Cluster                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Node 0     │    │   Node 1     │    │   Node 2     │       │
│  │  ┌────────┐  │    │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │ GPU 0  │  │    │  │ GPU 0  │  │    │  │ GPU 0  │  │       │
│  │  │ GPU 1  │  │    │  │ GPU 1  │  │    │  │ GPU 1  │  │       │
│  │  │  ...   │  │    │  │  ...   │  │    │  │  ...   │  │       │
│  │  │ GPU 7  │  │    │  │ GPU 7  │  │    │  │ GPU 7  │  │       │
│  │  └────────┘  │    │  └────────┘  │    │  └────────┘  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                   InfiniBand / RoCE Network                      │
│                    (400 Gbps per port)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Communication Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| All-Reduce | Aggregate + broadcast gradients | Data parallel training |
| All-Gather | Gather tensors from all ranks | Tensor parallel forward |
| Reduce-Scatter | Reduce + scatter results | ZeRO optimizer |
| Point-to-Point | Direct rank communication | Pipeline parallel |
| Broadcast | One-to-all distribution | Parameter sharing |

### NCCL Deep Dive

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(
    backend="nccl",
    init_method="env://",  # Uses MASTER_ADDR, MASTER_PORT
    world_size=16,         # Total GPUs across nodes
    rank=0                 # This process's rank
)

# All-reduce operation
tensor = torch.randn(1024, 1024, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# All-gather operation
gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered, tensor)

# Reduce-scatter
input_list = [torch.randn(1024, 1024, device='cuda') for _ in range(world_size)]
output = torch.zeros(1024, 1024, device='cuda')
dist.reduce_scatter(output, input_list)
```

### NCCL Environment Variables

```bash
# Network interface selection
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0

# Performance tuning
export NCCL_ALGO=Ring           # Ring, Tree, CollnetDirect
export NCCL_PROTO=LL128         # LL, LL128, Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8

# Debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# InfiniBand settings
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5     # GPU Direct RDMA level
```

### Ring vs Tree All-Reduce

**Ring All-Reduce:**
```
Step 1: Each GPU sends to next, receives from previous
Step 2: Repeat for n-1 iterations
Step 3: All GPUs have complete sum

GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0
  ↑                                ↓
  └────────────────────────────────┘
```

**Tree All-Reduce:**
```
        GPU 0 (root)
       /           \
    GPU 1         GPU 2
   /     \       /     \
GPU 3   GPU 4  GPU 5   GPU 6
```

| Algorithm | Latency | Bandwidth | Best For |
|-----------|---------|-----------|----------|
| Ring | O(n) | Optimal | Large messages |
| Tree | O(log n) | Sub-optimal | Small messages, low latency |
| Hybrid | Balanced | Good | General use |

---

## InfiniBand and GPU Direct

### InfiniBand Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    InfiniBand Network                           │
│                                                                  │
│   ┌─────────┐                              ┌─────────┐          │
│   │ Switch  │──────── 400 Gbps ───────────│ Switch  │          │
│   └────┬────┘                              └────┬────┘          │
│        │                                        │               │
│   ┌────┴────┐                              ┌────┴────┐          │
│   │   HCA   │                              │   HCA   │          │
│   │ (mlx5)  │                              │ (mlx5)  │          │
│   └────┬────┘                              └────┬────┘          │
│        │ PCIe                                   │ PCIe          │
│   ┌────┴────┐                              ┌────┴────┐          │
│   │   GPU   │                              │   GPU   │          │
│   └─────────┘                              └─────────┘          │
│                                                                  │
│   Node 0                                   Node 1               │
└─────────────────────────────────────────────────────────────────┘
```

### GPU Direct RDMA

Enables direct GPU-to-GPU communication over InfiniBand without CPU involvement.

```python
# Check if GPUDirect RDMA is available
import torch

# Environment setup for GDR
os.environ["NCCL_NET_GDR_LEVEL"] = "5"  # Enable GPUDirect
os.environ["NCCL_NET_GDR_READ"] = "1"   # Enable GDR for reads

# Verify with NCCL
# In logs, look for: "NET/IB : Using [0]mlx5_0:1/RoCE ; GDR[R/W]"
```

### GPUDirect Storage

Direct data path from storage to GPU memory.

```python
import cufile

# Open file with GPUDirect Storage
f = cufile.open("data.bin", "r")

# Allocate GPU buffer
gpu_buffer = torch.empty(1024, 1024, device='cuda')

# Direct read to GPU (bypasses CPU)
cufile.read(f, gpu_buffer.data_ptr(), gpu_buffer.nbytes)
```

---

## Pipeline Parallelism Details

### Pipeline Stages and Bubbles

```
Time →
        │ Stage 0 │ Stage 1 │ Stage 2 │ Stage 3 │
────────┼─────────┼─────────┼─────────┼─────────┤
Batch 0 │   F0    │   F0    │   F0    │  F0/B0  │
Batch 1 │   F1    │   F1    │  F1/B1  │   B0    │
Batch 2 │   F2    │  F2/B2  │   B1    │   B0    │
Batch 3 │  F3/B3  │   B2    │   B1    │   B0    │
────────┼─────────┼─────────┼─────────┼─────────┤
        │ Bubble  │ Bubble  │ Bubble  │         │

F = Forward pass, B = Backward pass
```

### Bubble Calculation

```
Pipeline bubble ratio = (p - 1) / (m + p - 1)

Where:
- p = number of pipeline stages
- m = number of micro-batches

Example:
- 4 stages, 8 micro-batches: (4-1)/(8+4-1) = 3/11 = 27% bubble
- 4 stages, 16 micro-batches: (4-1)/(16+4-1) = 3/19 = 16% bubble
```

### 1F1B Schedule

Interleaved forward and backward passes to reduce memory.

```python
# 1F1B (One Forward One Backward) schedule
def pipeline_1f1b(model_chunks, micro_batches, num_stages):
    # Warmup: fill pipeline
    for i in range(num_stages):
        forward(model_chunks[i], micro_batches[i])

    # Steady state: 1F1B
    for i in range(len(micro_batches) - num_stages):
        backward(model_chunks[num_stages-1], micro_batches[i])
        forward(model_chunks[0], micro_batches[num_stages + i])

    # Cooldown: drain pipeline
    for i in range(num_stages):
        backward(model_chunks[num_stages-1-i], micro_batches[-(i+1)])
```

### Virtual Pipeline Parallelism

Split model into more chunks than physical stages for better efficiency.

```
Physical stages: 4 GPUs
Virtual stages: 8 (2 per GPU)

GPU 0: [Layer 0-3, Layer 16-19]
GPU 1: [Layer 4-7, Layer 20-23]
GPU 2: [Layer 8-11, Layer 24-27]
GPU 3: [Layer 12-15, Layer 28-31]
```

---

## Tensor Parallelism Details

### Column Parallel Linear

```
Input X: [batch, seq, hidden]

Weight W split by columns:
GPU 0: W[:, :hidden/2]
GPU 1: W[:, hidden/2:]

Output:
GPU 0: Y0 = X @ W0  → [batch, seq, hidden/2]
GPU 1: Y1 = X @ W1  → [batch, seq, hidden/2]

All-gather: Y = [Y0, Y1]  → [batch, seq, hidden]
```

### Row Parallel Linear

```
Input X split by last dimension:
GPU 0: X[:, :, :hidden/2]
GPU 1: X[:, :, hidden/2:]

Weight W split by rows:
GPU 0: W[:hidden/2, :]
GPU 1: W[hidden/2:, :]

Output:
GPU 0: Y0 = X0 @ W0
GPU 1: Y1 = X1 @ W1

All-reduce: Y = Y0 + Y1
```

### Attention Tensor Parallelism

```python
# Multi-head attention with tensor parallelism
class TensorParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, tp_size):
        self.num_heads_per_partition = num_heads // tp_size
        self.hidden_per_partition = hidden_size // tp_size

        # Column parallel for Q, K, V
        self.qkv = ColumnParallelLinear(
            hidden_size,
            3 * hidden_size,
            gather_output=False
        )

        # Row parallel for output projection
        self.output = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True
        )

    def forward(self, x):
        # QKV projection (column parallel)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention (local computation)
        attn_output = scaled_dot_product_attention(q, k, v)

        # Output projection (row parallel with all-reduce)
        output = self.output(attn_output)
        return output
```

---

## Sequence Parallelism

Split sequence dimension to reduce activation memory.

```
Without Sequence Parallelism:
Activations: [batch, seq_len, hidden] on each GPU

With Sequence Parallelism:
GPU 0: [batch, seq_len/tp, hidden]  ← First portion of sequence
GPU 1: [batch, seq_len/tp, hidden]  ← Second portion of sequence
...

Memory reduction: tp times less activation memory
```

```python
# NeMo configuration for sequence parallelism
model:
  tensor_model_parallel_size: 4
  sequence_parallel: true  # Enable SP alongside TP
```

---

## ZeRO (Zero Redundancy Optimizer)

### ZeRO Stages

| Stage | Partitions | Memory Savings | Communication |
|-------|------------|----------------|---------------|
| ZeRO-1 | Optimizer states | ~4x | Same as DP |
| ZeRO-2 | + Gradients | ~8x | +All-gather gradients |
| ZeRO-3 | + Parameters | ~N (linear) | +All-gather params |

### ZeRO-3 Configuration

```python
# DeepSpeed ZeRO-3 config
zero_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

### FSDP (Fully Sharded Data Parallel)

PyTorch native ZeRO-3 implementation.

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16
    ),
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # For compatibility with torch.compile
)
```

---

## DGX and SuperPOD Architecture

### DGX H100 Specifications

| Component | Specification |
|-----------|---------------|
| GPUs | 8x H100 80GB SXM5 |
| GPU Memory | 640 GB HBM3 total |
| NVLink | 900 GB/s per GPU |
| NVSwitch | 4th gen, 7.2 TB/s bisection |
| Network | 8x 400 Gbps InfiniBand |
| CPU | 2x Intel Xeon Platinum 8480C |
| System Memory | 2 TB DDR5 |
| Storage | 30 TB NVMe |

### NVSwitch Topology

```
                    ┌─────────────┐
                    │  NVSwitch   │
                    │   Fabric    │
                    └──────┬──────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
┌───┴───┐  ┌───────┐  ┌───┴───┐  ┌───────┐  ┌───┴───┐
│ GPU 0 │──│NVLink │──│ GPU 1 │──│NVLink │──│ GPU 2 │
└───────┘  └───────┘  └───────┘  └───────┘  └───────┘
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                    All-to-all connectivity
```

### SuperPOD Configuration

```
DGX SuperPOD:
├── 32 DGX H100 nodes
├── 256 H100 GPUs
├── 20 PB/s aggregate GPU memory bandwidth
├── InfiniBand NDR 400G fabric
└── Shared parallel file system (GPFS/Lustre)

Typical 3D parallelism config for 175B model:
- Tensor Parallel: 8 (within DGX)
- Pipeline Parallel: 4 (across DGX nodes)
- Data Parallel: 8 (across node groups)
- Total: 8 × 4 × 8 = 256 GPUs
```

---

## Debugging Multi-GPU Issues

### Common Problems and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| NCCL timeout | Hangs during collective | Check network, increase timeout |
| OOM one GPU | Single GPU runs out | Balance model/data partitioning |
| Slow communication | Low throughput | Check NVLink/IB bandwidth |
| Gradient mismatch | NaN/divergence | Check reduction ops, precision |
| Deadlock | Complete hang | Verify all ranks call same collectives |

### Debugging Commands

```bash
# Check GPU topology
nvidia-smi topo -m

# Check NVLink status
nvidia-smi nvlink -s

# Check InfiniBand
ibstat
ibv_devinfo

# Monitor GPU usage across nodes
dcgmi dmon -e 150,155,156,200,201,203,204

# NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# PyTorch distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### Verifying Communication

```python
import torch
import torch.distributed as dist

def verify_all_reduce():
    """Verify all-reduce works correctly."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each rank has tensor with its rank value
    tensor = torch.tensor([rank], dtype=torch.float32, device='cuda')

    # All-reduce should give sum of all ranks
    dist.all_reduce(tensor)

    expected = sum(range(world_size))
    assert tensor.item() == expected, f"All-reduce failed: got {tensor.item()}, expected {expected}"
    print(f"Rank {rank}: All-reduce verified successfully")
```

---

## Best Practices Summary

1. **Start with data parallel** - Simplest, then add complexity as needed
2. **Profile before optimizing** - Use Nsight to identify bottlenecks
3. **Maximize NVLink usage** - Keep tensor parallel within NVLink domain
4. **Tune micro-batch size** - Balance pipeline bubbles vs memory
5. **Use mixed precision** - BF16 for training, FP8 for inference
6. **Enable gradient checkpointing** - Trade compute for memory
7. **Optimize communication** - Overlap compute and communication
8. **Monitor utilization** - Target >80% GPU utilization
9. **Use appropriate parallelism** - Match strategy to model/hardware
10. **Test at scale early** - Multi-node bugs appear at scale
