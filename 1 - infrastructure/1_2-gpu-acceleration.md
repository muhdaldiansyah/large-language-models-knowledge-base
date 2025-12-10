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
