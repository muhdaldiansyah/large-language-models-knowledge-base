# Group A - Infrastructure (40% weight)

## A1. Model Optimization (17%)

### Quantization Methods
- FP8 (E4M3, E5M2 formats)
- INT8 SmoothQuant (activation smoothing)
- INT4 AWQ / GPTQ (weight-only)
- W4A8 (4-bit weights, 8-bit activations)

### Key Concepts
- AMAX calibration (dynamic vs static)
- KV-cache quantization
- In-flight batching
- PagedAttention (vLLM-style memory management)
- PTQ vs QAT (Post-Training vs Quantization-Aware Training)

### Mixed Precision
- FP16, BF16 (brain float)
- AMP O0/O1/O2 optimization levels
- Loss scaling for mixed precision

### Compute Estimation
- FLOPs calculation: `6 * N * D` per token (forward + backward)
- Activation memory formulas
- Tokens/sec throughput calculations

**Detailed notes:** [1_1-model-optimization.md](1_1-model-optimization.md)

---

## A2. GPU Acceleration & Optimization (14%)

### Parallelism Strategies
- **Tensor Parallel (TP)**: Split layers across GPUs
- **Pipeline Parallel (PP)**: Split model stages across GPUs
- **Data Parallel (DP)**: Replicate model, split data
- **Sequence Parallelism (SP)**: Split sequence dimension

### When to Use Each
| Strategy | Best For | Communication |
|----------|----------|---------------|
| TP | Large layers, low latency | High (all-reduce) |
| PP | Very deep models | Low (point-to-point) |
| DP | Large batch training | Medium (gradient sync) |

### Nsight Metrics
- SM Occupancy (GPU utilization)
- Kernel Launch Latency
- Memory Bandwidth utilization
- Compute vs Memory Bound analysis

### Hardware Interconnects
- **NVLink**: High-bandwidth GPU-to-GPU (900 GB/s on H100)
- **PCIe**: Standard interconnect (~64 GB/s Gen5)
- **NCCL**: NVIDIA Collective Communications Library

### Hardware Architecture
- Tensor Cores (matrix operations)
- DGX Architecture (8x GPUs, NVSwitch)
- HBM3 memory bandwidth

**Detailed notes:** [1_2-gpu-acceleration.md](1_2-gpu-acceleration.md)

---

## A3. Model Deployment (9%)

### Triton Inference Server
- Model Repository structure
- config.pbtxt configuration
- Multi-model serving

### Key Configuration Parameters
```
max_batch_size: 64
instance_group: [{count: 2, kind: KIND_GPU}]
dynamic_batching:
  preferred_batch_size: [16, 32]
  max_queue_delay_microseconds: 100
```

### Orchestration Platforms
- **Kubernetes**: Production orchestration, horizontal scaling
- **Docker / NGC**: Container-based deployment
- **Slurm**: HPC job scheduling (srun, sbatch)

### TensorRT-LLM Runtime
- Static vs dynamic shapes
- CUDA graph capture (reduces kernel launch overhead)
- Layer fusion optimizations
- Continuous batching

**Detailed notes:** [1_3-deployment.md](1_3-deployment.md)

---

## A4. NVIDIA NIM

### Overview
- Pre-optimized inference microservices
- Day-1 deployment capability
- OpenAI-compatible API endpoints

### Key Features
- Automatic hardware optimization
- Built-in scaling and load balancing
- Enterprise support and security

### Deployment
```bash
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

**Detailed notes:** [1_4-nvidia-nim.md](1_4-nvidia-nim.md)

---

## Quick Reference

| Topic | Weight | Key Technologies |
|-------|--------|------------------|
| Model Optimization | 17% | TensorRT-LLM, FP8, INT8, PagedAttention |
| GPU Acceleration | 14% | TP/PP/DP, NVLink, NCCL, Nsight |
| Deployment | 9% | Triton, Kubernetes, dynamic batching |
| NVIDIA NIM | - | OpenAI API, NGC containers |
