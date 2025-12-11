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

---

## Batching Strategies

### Static Batching

Traditional approach - wait for fixed batch before processing.

```
Requests: [A, B, C, D] → Wait for batch → Process all → Return all
                            ↓
                    All requests same length
                    Padding for shorter ones
```

**Problems:**
- Wasted compute on padding
- All requests must complete together
- Poor GPU utilization for variable-length outputs

### Continuous Batching (In-flight Batching)

Process requests as slots become available.

```
Time →
Slot 1: [Request A generating........] [Request D generating...]
Slot 2: [Request B done] [Request E generating........]
Slot 3: [Request C generating...] [Request F generating....]
                    ↓
         New requests join immediately when slots free
```

**Benefits:**
- No padding waste
- Requests return immediately when done
- 2-10x throughput improvement
- Better GPU utilization

### TensorRT-LLM Continuous Batching

```python
# Enable in-flight batching
trtllm-build \
    --model llama-70b \
    --use_inflight_batching \
    --paged_kv_cache \
    --max_batch_size 256 \
    --max_num_tokens 8192
```

### Triton + TensorRT-LLM Config

```protobuf
# config.pbtxt for in-flight batching
name: "llama"
backend: "tensorrtllm"
max_batch_size: 256

model_transaction_policy {
  decoupled: true  # Required for streaming
}

dynamic_batching {
  max_queue_delay_microseconds: 1000
}

parameters {
  key: "batching_type"
  value: { string_value: "inflight_fused_batching" }
}
```

---

## vLLM

High-throughput LLM serving with PagedAttention.

### Key Features

- **PagedAttention**: Efficient KV cache memory management
- **Continuous Batching**: Dynamic request scheduling
- **Prefix Caching**: Reuse KV cache for shared prefixes
- **Speculative Decoding**: Draft model acceleration

### PagedAttention

```
Traditional KV Cache:
Request 1: [=================----] (contiguous, with padding)
Request 2: [============---------] (contiguous, with padding)

PagedAttention:
Request 1: [Page1][Page2][Page3][Page4]
Request 2: [Page5][Page6][Page3]  ← Can share pages!
                           ↑
              Pages allocated on demand, no padding
```

**Benefits:**
- Near-zero memory waste
- Efficient memory sharing
- 2-4x more concurrent requests

### vLLM Usage

```python
from vllm import LLM, SamplingParams

# Initialize
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=8,
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=8192
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Generate
outputs = llm.generate(prompts, sampling_params)
```

### vLLM Server

```bash
# Start OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 4096 \
    --port 8000

# Use with OpenAI client
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello"}]}'
```

### vLLM vs TensorRT-LLM

| Feature | vLLM | TensorRT-LLM |
|---------|------|--------------|
| Ease of use | Easy | Complex |
| Optimization | Good | Best |
| Latency | Good | Best |
| Throughput | Excellent | Excellent |
| Model support | Wide | Growing |
| Customization | Python | C++/Python |

---

## TensorRT-LLM Detailed

### Build Process

```bash
# 1. Convert model to TensorRT-LLM format
python convert_checkpoint.py \
    --model_dir /models/llama-70b \
    --output_dir /engines/llama-70b-ckpt \
    --dtype float16

# 2. Build TensorRT engine
trtllm-build \
    --checkpoint_dir /engines/llama-70b-ckpt \
    --output_dir /engines/llama-70b-engine \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --paged_kv_cache enable \
    --use_inflight_batching \
    --use_fp8_context_fmha enable
```

### Key Build Flags

| Flag | Purpose |
|------|---------|
| `--gemm_plugin` | Optimized matrix multiply |
| `--gpt_attention_plugin` | Fused attention kernels |
| `--paged_kv_cache` | Memory-efficient KV cache |
| `--use_inflight_batching` | Continuous batching |
| `--use_fp8_context_fmha` | FP8 attention (H100) |
| `--max_num_tokens` | Max concurrent tokens |
| `--tp_size` | Tensor parallelism degree |
| `--pp_size` | Pipeline parallelism degree |

### Quantization in Build

```bash
# FP8 quantization (H100)
trtllm-build \
    --checkpoint_dir ./ckpt \
    --output_dir ./engine \
    --use_fp8 \
    --fp8_kv_cache

# INT8 SmoothQuant
trtllm-build \
    --checkpoint_dir ./ckpt \
    --output_dir ./engine \
    --use_smooth_quant \
    --per_channel \
    --per_token

# INT4 AWQ
trtllm-build \
    --checkpoint_dir ./ckpt \
    --output_dir ./engine \
    --use_weight_only \
    --weight_only_precision int4 \
    --use_awq
```

### Runtime Execution

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

# Load engine
runner = ModelRunner.from_dir("/engines/llama-70b-engine")

# Generate
outputs = runner.generate(
    batch_input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    streaming=True
)
```

---

## ONNX Conversion

### Export to ONNX

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("llama-7b")

# Export to ONNX
dummy_input = torch.randint(0, 32000, (1, 512))

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=17
)
```

### Optimize ONNX

```python
from onnxruntime.transformers import optimizer

# Optimize for inference
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type="gpt2",  # or "bert", "t5"
    num_heads=32,
    hidden_size=4096,
    use_gpu=True
)

optimized_model.save_model_to_file("model_optimized.onnx")
```

### ONNX Runtime Inference

```python
import onnxruntime as ort

# Create session with GPU
session = ort.InferenceSession(
    "model_optimized.onnx",
    providers=["CUDAExecutionProvider"]
)

# Run inference
outputs = session.run(
    None,
    {"input_ids": input_ids.numpy()}
)
```

---

## Model Serving Comparison

| Platform | Latency | Throughput | Ease | Best For |
|----------|---------|------------|------|----------|
| TensorRT-LLM + Triton | Best | Highest | Complex | Production NVIDIA |
| vLLM | Good | High | Easy | Quick deployment |
| NVIDIA NIM | Good | High | Easiest | Enterprise |
| ONNX Runtime | Medium | Medium | Easy | Portability |
| HuggingFace TGI | Good | Good | Easy | HF ecosystem |

---

## Deployment Best Practices

### Capacity Planning

```python
def estimate_serving_capacity(
    model_size_gb,
    gpu_memory_gb,
    avg_input_len,
    avg_output_len,
    concurrent_users
):
    """Estimate GPUs needed for serving"""

    # KV cache per request
    kv_per_token = 0.5  # MB, rough estimate
    kv_per_request = (avg_input_len + avg_output_len) * kv_per_token / 1024  # GB

    # Memory per GPU
    available = gpu_memory_gb - model_size_gb
    requests_per_gpu = available / kv_per_request

    # Total GPUs needed
    gpus_needed = concurrent_users / requests_per_gpu

    return math.ceil(gpus_needed)
```

### Autoscaling Configuration

```yaml
# Kubernetes HPA for LLM service
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
  - type: Pods
    pods:
      metric:
        name: request_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

### Load Balancing for LLMs

```
Considerations for LLM load balancing:
1. Session affinity - keep streaming connections on same backend
2. Health checks - verify model is loaded and responsive
3. Queue depth - route to least loaded instance
4. GPU memory - avoid overloading specific instances
```

---

## Troubleshooting Deployment

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Slow first request | High TTFT initially | Model warmup, preload |
| Memory leak | OOM over time | Check KV cache cleanup, restart |
| Inconsistent latency | Variable response times | Enable CUDA graphs, check batching |
| Low throughput | GPU underutilized | Increase batch size, enable continuous batching |
| Model loading slow | Long startup | Use shared memory, faster storage |
| Connection timeout | Requests fail | Increase timeout, check load balancer |
