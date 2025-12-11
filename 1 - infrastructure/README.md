# Group A - Infrastructure (40% weight)

## A1. Model Optimization (17%)

### Quantization Methods
- FP8 (Hopper architecture)
- INT8 SmoothQuant
- INT4 AWQ / GPTQ
- W4A8 hybrid

### Key Concepts
- AMAX calibration
- KV-cache quantization
- In-flight batching
- PagedAttention
- PTQ vs QAT

### Pruning
- Unstructured pruning (magnitude-based)
- Structured pruning (heads, layers)
- 2:4 sparsity (NVIDIA ASP)
- SparseGPT

### Knowledge Distillation
- Response-based distillation
- Feature-based distillation
- Temperature and alpha parameters

### Speculative Decoding
- Draft model approach
- Medusa multi-head speculation
- Performance characteristics

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

### Memory Optimization
- Gradient checkpointing
- Activation checkpointing
- FlashAttention (v1/v2)
- Memory calculation formulas

### NCCL Optimization
- Collective operations (AllReduce, AllGather)
- Environment variables tuning
- Multi-node configuration

### Nsight Metrics
- SM Occupancy
- Kernel Launch Latency
- Memory Bandwidth
- Compute vs Memory Bound

### Hardware
- NVLink vs PCIe
- Tensor Cores
- DGX Architecture
- GPU benchmarks

---

## A3. Model Deployment (9%)

### Triton Inference Server
- Model Repository structure
- config.pbtxt configuration

### Batching Strategies
- Static batching
- Continuous batching (in-flight)
- PagedAttention

### vLLM
- PagedAttention memory management
- Server deployment
- Comparison with TensorRT-LLM

### TensorRT-LLM
- Build process and flags
- Quantization options
- Runtime execution

### ONNX Conversion
- Export workflow
- Optimization
- Runtime inference

### Key Parameters
- max_batch_size
- instance_group
- dynamic_batching
- preferred_batch_size
- max_queue_delay_microseconds

### Orchestration
- Kubernetes + HPA
- Docker / NGC
- Slurm (srun, sbatch)

---

## A4. NVIDIA NIM

- Definition and purpose
- Day-1 Deployment
- OpenAI-compatible API

---

## A5. NVIDIA AI Ecosystem

### NGC Catalog
- Container registry
- Model downloads
- CLI usage

### TAO Toolkit
- Fine-tuning workflow
- Pruning and quantization
- Export formats

### Base Command Platform
- Job submission
- Instance types
- Dataset management

### NVIDIA AI Enterprise
- Components overview
- Licensing tiers
- Deployment options

### NeMo Curator
- Data curation pipeline
- GPU-accelerated processing
- Deduplication and filtering
