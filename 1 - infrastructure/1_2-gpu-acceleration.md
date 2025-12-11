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
