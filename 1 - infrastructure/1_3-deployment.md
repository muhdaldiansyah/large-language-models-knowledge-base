# 1.3 Model Deployment (9%)

## Triton Inference Server

### Model Repository Structure
```
model_repository/
├── model_name/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.plan
│   └── 2/
│       └── model.plan
```

### config.pbtxt
```
name: "model_name"
backend: "tensorrtllm"
max_batch_size: 64

input [
  { name: "input_ids" data_type: TYPE_INT32 dims: [-1] }
]

output [
  { name: "output_ids" data_type: TYPE_INT32 dims: [-1] }
]
```

---

## Key Parameters

### max_batch_size
- Maximum requests processed together
- Higher = better throughput
- Limited by GPU memory

### instance_group
- Number of model instances per GPU
- `count`: instances per device
- `kind`: GPU or CPU

### dynamic_batching
- Groups requests automatically
- Reduces latency variance
- Improves GPU utilization

### preferred_batch_size
- Target batch sizes for optimization
- Example: [4, 8, 16, 32]

### max_queue_delay_microseconds
- Maximum wait time for batching
- Tradeoff: latency vs throughput
- Example: 100000 (100ms)

---

## Ensembling

- Chain multiple models in pipeline
- Preprocessing → Model → Postprocessing
- Defined in config.pbtxt
- Shared memory between stages

---

## Orchestration

### Kubernetes
- Container orchestration
- Horizontal pod autoscaling
- GPU resource scheduling
- NVIDIA GPU Operator

### Docker / NGC
- NGC containers for NVIDIA software
- Pre-optimized images
- `nvcr.io/nvidia/tritonserver`
- `nvcr.io/nvidia/tensorrt`

### Slurm
- HPC job scheduler
- `srun` - interactive jobs
- `sbatch` - batch jobs
- GPU allocation with `--gres=gpu:N`

---

## TensorRT-LLM Runtime

### Static vs Dynamic Shapes
- **Static**: Fixed input dimensions, faster
- **Dynamic**: Variable dimensions, flexible
- Use dynamic for LLM inference

### CUDA Graph Capture
- Records kernel launches
- Reduces CPU overhead
- Best for repeated operations
- Enable with `--use_cuda_graph`

### Layer Fusion
- Combines multiple operations
- Reduces memory bandwidth
- Examples: QKV fusion, MLP fusion
- Automatic in TensorRT optimization
