# Group A - Infrastructure

## A1. Model Optimization (17%)

### Quantization Methods
- FP8
- INT8 SmoothQuant
- INT4 AWQ / GPTQ
- W4A8

### Key Concepts
- AMAX calibration
- KV-cache quantization
- In-flight batching
- PagedAttention
- PTQ vs QAT

### Mixed Precision
- FP16, BF16
- AMP O0/O1/O2

### Compute Estimation
- FLOPs calculation
- Activation memory
- Tokens/sec formulas

---

## A2. GPU Acceleration & Optimization (14%)

### Parallelism Strategies
- Tensor Parallel (TP)
- Pipeline Parallel (PP)
- Data Parallel (DP)
- Sequence Parallelism (SP)

### Nsight Metrics
- SM Occupancy
- Kernel Launch Latency
- Memory Bandwidth
- Compute vs Memory Bound

### Hardware
- NVLink vs PCIe
- NCCL
- Tensor Cores
- DGX Architecture

---

## A3. Model Deployment (9%)

### Triton Inference Server
- Model Repository structure
- config.pbtxt configuration

### Key Parameters
- max_batch_size
- instance_group
- dynamic_batching
- preferred_batch_size
- max_queue_delay_microseconds

### Orchestration
- Kubernetes
- Docker / NGC
- Slurm (srun, sbatch)

### TensorRT-LLM Runtime
- Static vs dynamic shapes
- CUDA graph capture
- Layer fusion

---

## A4. NVIDIA NIM

- Definition and purpose
- Day-1 Deployment
- OpenAI-compatible API
