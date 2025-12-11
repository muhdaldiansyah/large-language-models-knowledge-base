# 1.3 Model Deployment (9%)

## Overview

Model deployment involves taking trained models and serving them in production environments with high availability, low latency, and efficient resource utilization.

```
Training → Optimization → Containerization → Orchestration → Serving → Monitoring
              ↓                                    ↓
         TensorRT-LLM                         Kubernetes
```

---

## Triton Inference Server

NVIDIA Triton is an open-source inference serving platform supporting multiple frameworks and optimized for NVIDIA GPUs.

### Architecture

```
                    ┌─────────────────────────────────┐
                    │         Triton Server           │
                    │  ┌───────────────────────────┐  │
 HTTP/gRPC ────────►│  │    Request Handler       │  │
                    │  └───────────────────────────┘  │
                    │  ┌───────────────────────────┐  │
                    │  │    Dynamic Batcher       │  │
                    │  └───────────────────────────┘  │
                    │  ┌───────────────────────────┐  │
                    │  │    Model Scheduler       │  │
                    │  └───────────────────────────┘  │
                    │  ┌─────┐ ┌─────┐ ┌─────────┐  │
                    │  │TRT- │ │vLLM │ │Python   │  │
                    │  │LLM  │ │     │ │Backend  │  │
                    │  └─────┘ └─────┘ └─────────┘  │
                    └─────────────────────────────────┘
```

### Model Repository Structure

```
model_repository/
├── model_name/
│   ├── config.pbtxt          # Model configuration
│   ├── 1/                    # Version 1
│   │   └── model.plan        # TensorRT engine
│   ├── 2/                    # Version 2
│   │   └── model.plan
│   └── labels.txt            # Optional labels
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py          # Python backend
└── ensemble/
    └── config.pbtxt          # Pipeline definition
```

### config.pbtxt - Complete Reference

```protobuf
name: "llama3-8b"
backend: "tensorrtllm"
max_batch_size: 64

# Model versioning
version_policy: { latest: { num_versions: 2 } }

# Input specification
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1]  # Dynamic sequence length
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [1]
    reshape: { shape: [] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [1]
    reshape: { shape: [] }
  }
]

# Output specification
output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  },
  {
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [-1]
  }
]

# Instance configuration
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [4, 8, 16, 32]
  max_queue_delay_microseconds: 100000
  preserve_ordering: false
  priority_levels: 2
  default_priority_level: 1
}

# Response cache
response_cache {
  enable: true
}

# Rate limiting
rate_limiter {
  resources [
    { name: "gpu_memory", count: 1 }
  ]
}

# Model warmup
model_warmup [
  {
    name: "warmup_request"
    batch_size: 1
    inputs {
      key: "input_ids"
      value: { dims: [32], data_type: TYPE_INT32 }
    }
  }
]

# Parameters for TensorRT-LLM backend
parameters {
  key: "gpt_model_type"
  value: { string_value: "llama" }
}
parameters {
  key: "gpt_model_path"
  value: { string_value: "/models/llama3-8b/engine" }
}
parameters {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "2560" }
}
parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "inflight_fused_batching" }
}
```

---

## Key Deployment Parameters

### Batching Configuration

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `max_batch_size` | Maximum requests per batch | 32-128 |
| `preferred_batch_size` | Optimal batch sizes | [8, 16, 32] |
| `max_queue_delay_microseconds` | Max wait for batching | 100000 (100ms) |
| `preserve_ordering` | Maintain request order | false |
| `priority_levels` | Number of priority queues | 2-3 |

### Instance Groups

```protobuf
# Single GPU, single instance
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] }
]

# Multi-GPU with tensor parallelism
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0, 1, 2, 3] }
]

# Multiple instances across GPUs
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] },
  { count: 2, kind: KIND_GPU, gpus: [1] }
]

# CPU fallback
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] },
  { count: 2, kind: KIND_CPU }
]
```

### Response Cache

Caches responses for identical requests:

```protobuf
response_cache {
  enable: true
}
```

**Benefits:**
- Reduces redundant computation
- Lower latency for repeated queries
- Automatic cache invalidation on model update

### Rate Limiting

Control resource consumption:

```protobuf
rate_limiter {
  resources [
    { name: "gpu_memory", count: 8 },
    { name: "compute", count: 4 }
  ]
}
```

---

## Continuous Batching (In-Flight Batching)

Traditional batching waits for batch to complete. Continuous batching adds new requests as tokens complete.

```
Traditional Batching:
Request 1: [████████████████████]
Request 2: [████████████████████]  ← Waits for longest
Request 3: [████████]              ← Finished early, waits

Continuous Batching:
Request 1: [████████████████████]
Request 2: [████████████]→[New Req 4]
Request 3: [████████]→[New Req 5]→[New Req 6]
           ↑ New requests added immediately
```

**Enable in TensorRT-LLM:**
```protobuf
parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "inflight_fused_batching" }
}
parameters {
  key: "enable_chunked_context"
  value: { string_value: "true" }
}
```

**Benefits:**
- 2-3x higher throughput
- Lower average latency
- Better GPU utilization

---

## Model Versioning

### Version Policy Options

```protobuf
# Serve latest N versions
version_policy: { latest: { num_versions: 2 } }

# Serve all versions
version_policy: { all: {} }

# Serve specific versions
version_policy: { specific: { versions: [1, 3, 5] } }
```

### Version Selection in Request

```python
# Request specific version
triton_client.infer(
    model_name="llama3",
    model_version="2",  # Specific version
    inputs=inputs
)
```

---

## Ensemble Pipelines

Chain multiple models for complex workflows:

```protobuf
# ensemble/config.pbtxt
name: "rag_pipeline"
platform: "ensemble"
max_batch_size: 8

input [
  { name: "query", data_type: TYPE_STRING, dims: [1] }
]

output [
  { name: "response", data_type: TYPE_STRING, dims: [1] }
]

ensemble_scheduling {
  step [
    {
      model_name: "embedder"
      model_version: -1
      input_map { key: "text", value: "query" }
      output_map { key: "embedding", value: "query_embedding" }
    },
    {
      model_name: "retriever"
      model_version: -1
      input_map { key: "query_embedding", value: "query_embedding" }
      output_map { key: "documents", value: "context" }
    },
    {
      model_name: "llm"
      model_version: -1
      input_map {
        key: "prompt"
        value: "query"
      }
      input_map {
        key: "context"
        value: "context"
      }
      output_map { key: "response", value: "response" }
    }
  ]
}
```

---

## Kubernetes Deployment

### GPU Resource Configuration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-llm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: triton-llm
  template:
    metadata:
      labels:
        app: triton-llm
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
        ports:
        - containerPort: 8000  # HTTP
        - containerPort: 8001  # gRPC
        - containerPort: 8002  # Metrics
        resources:
          limits:
            nvidia.com/gpu: 4          # Request 4 GPUs
            memory: "64Gi"
            cpu: "16"
          requests:
            nvidia.com/gpu: 4
            memory: "32Gi"
            cpu: "8"
        volumeMounts:
        - name: model-repository
          mountPath: /models
        - name: shm
          mountPath: /dev/shm
        args:
        - tritonserver
        - --model-repository=/models
        - --log-verbose=1
        - --strict-model-config=false
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: model-repository
        persistentVolumeClaim:
          claimName: model-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-llm
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: triton_queue_size
      target:
        type: AverageValue
        averageValue: "10"
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: triton-service
spec:
  selector:
    app: triton-llm
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: grpc
    port: 8001
    targetPort: 8001
  - name: metrics
    port: 8002
    targetPort: 8002
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: triton-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: triton-service
            port:
              number: 8000
```

### NVIDIA GPU Operator

```bash
# Install GPU Operator via Helm
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set mig.strategy=single
```

---

## Docker / NGC Deployment

### Running Triton Container

```bash
# Pull Triton with TensorRT-LLM backend
docker pull nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3

# Run with model repository
docker run --gpus all -d \
  --shm-size=16g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=1 \
    --strict-model-config=false
```

### Docker Compose for Multi-Service

```yaml
# docker-compose.yml
version: '3.8'

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '16gb'
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./models:/models
      - ./plugins:/plugins
    command: >
      tritonserver
      --model-repository=/models
      --log-verbose=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Slurm HPC Deployment

### Batch Job Script

```bash
#!/bin/bash
#SBATCH --job-name=triton-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=triton_%j.log

# Load modules
module load cuda/12.2
module load singularity

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run Triton in Singularity container
singularity run --nv \
  --bind /path/to/models:/models \
  tritonserver_24.05.sif \
  tritonserver \
    --model-repository=/models \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
```

### Multi-Node Distributed Inference

```bash
#!/bin/bash
#SBATCH --job-name=distributed-llm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00

# Get node list
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Launch on each node
srun --ntasks-per-node=1 \
  python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    serve_model.py
```

---

## TensorRT-LLM Runtime

### Building TensorRT-LLM Engine

```bash
# Convert model to TensorRT-LLM format
python convert_checkpoint.py \
  --model_dir ./llama-3-8b \
  --output_dir ./trt_ckpt \
  --dtype float16 \
  --tp_size 4 \
  --pp_size 1

# Build engine
trtllm-build \
  --checkpoint_dir ./trt_ckpt \
  --output_dir ./trt_engine \
  --gemm_plugin float16 \
  --gpt_attention_plugin float16 \
  --max_batch_size 64 \
  --max_input_len 2048 \
  --max_output_len 512 \
  --max_num_tokens 8192 \
  --use_paged_context_fmha enable \
  --use_fp8_context_fmha enable \
  --multiple_profiles enable
```

### Runtime Configuration

```python
# TensorRT-LLM Python runtime
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="./trt_engine",
    rank=0,
    debug_mode=False
)

outputs = runner.generate(
    batch_input_ids=input_ids,
    max_new_tokens=256,
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    streaming=True
)
```

### Static vs Dynamic Shapes

| Aspect | Static | Dynamic |
|--------|--------|---------|
| Input size | Fixed | Variable |
| Performance | Faster | Slightly slower |
| Flexibility | Limited | High |
| Memory | Predictable | Variable |
| Use case | Fixed-length tasks | Chat/completion |

### CUDA Graph Capture

```python
# Enable CUDA graphs for reduced CPU overhead
runner = ModelRunner.from_dir(
    engine_dir="./trt_engine",
    use_cuda_graph=True,
    cuda_graph_cache_size=2  # Cache N graph configurations
)
```

**When to use:**
- Consistent batch sizes
- Fixed sequence lengths
- Low-latency requirements

### Layer Fusion Optimizations

TensorRT automatically fuses operations:

| Fusion | Operations Combined | Benefit |
|--------|---------------------|---------|
| QKV Fusion | Q, K, V projections | 3x fewer kernels |
| MLP Fusion | Gate + Up + Down | Reduced memory |
| Attention Fusion | Softmax + MatMul | FlashAttention |
| LayerNorm Fusion | Norm + residual | Single kernel |

---

## vLLM Integration

### vLLM as Alternative Backend

```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    tensor_parallel_size=4,
    dtype="float16",
    max_model_len=8192,
    gpu_memory_utilization=0.9
)

# Generate
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)
outputs = llm.generate(prompts, sampling_params)
```

### vLLM OpenAI-Compatible Server

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8b-instruct \
  --tensor-parallel-size 4 \
  --port 8000
```

### TensorRT-LLM vs vLLM

| Aspect | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Optimization | Highest | High |
| Ease of use | Moderate | Easy |
| Flexibility | Lower | Higher |
| GPU support | NVIDIA only | NVIDIA + AMD |
| Latency | Best | Very good |
| Throughput | Best | Very good |

---

## Load Balancing & Scaling

### Request Routing Strategies

```python
# Round-robin load balancing
from itertools import cycle

servers = ["server1:8000", "server2:8000", "server3:8000"]
server_cycle = cycle(servers)

def route_request(request):
    return next(server_cycle)

# Least-connections
import heapq

def route_least_connections(servers, connections):
    return heapq.heappop(connections)

# Consistent hashing (for cache locality)
import hashlib

def route_consistent_hash(request, servers):
    key = hashlib.md5(request.prompt.encode()).hexdigest()
    index = int(key, 16) % len(servers)
    return servers[index]
```

### Blue-Green Deployment

```yaml
# Blue deployment (current)
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm
    version: blue  # Points to blue
---
# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-green
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: llm
        version: green
```

### Canary Deployment

```yaml
# Istio VirtualService for canary
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-canary
spec:
  hosts:
  - llm-service
  http:
  - route:
    - destination:
        host: llm-service
        subset: stable
      weight: 90
    - destination:
        host: llm-service
        subset: canary
      weight: 10
```

---

## Production Checklist

### Pre-Deployment
- [ ] Model optimized (TensorRT-LLM engine built)
- [ ] Batch size tuned for target latency
- [ ] Memory requirements calculated
- [ ] Load testing completed
- [ ] Fallback strategy defined

### Deployment Configuration
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Autoscaling policies defined
- [ ] Logging enabled
- [ ] Metrics exported

### Post-Deployment
- [ ] Monitoring dashboards active
- [ ] Alerting rules configured
- [ ] Runbooks documented
- [ ] Rollback procedure tested

---

## Best Practices

1. **Always use continuous batching** for LLM workloads
2. **Set appropriate timeouts** - LLM inference can be slow
3. **Use shared memory** (`/dev/shm`) for inter-process communication
4. **Monitor queue depth** - indicates capacity issues
5. **Implement circuit breakers** for downstream failures
6. **Use model warmup** to avoid cold-start latency
7. **Enable response caching** for repeated queries
8. **Set GPU memory limits** to prevent OOM
9. **Use node selectors** to target appropriate GPU types
10. **Implement graceful shutdown** for zero-downtime updates
