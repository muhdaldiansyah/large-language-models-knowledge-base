# 1.4 NVIDIA NIM (NVIDIA Inference Microservices)

## Overview

NVIDIA NIM is a set of optimized inference microservices that enable deployment of AI models with minimal setup. NIMs are pre-built, optimized containers that abstract away the complexity of model optimization and deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                      NVIDIA NIM                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              OpenAI-Compatible API                   │   │
│  │         /v1/chat/completions, /v1/embeddings        │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Automatic Optimization Layer               │   │
│  │    (TensorRT-LLM, Quantization, Batching)           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Triton Inference Server                 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  CUDA / cuDNN                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Value Propositions

| Feature | Benefit |
|---------|---------|
| Pre-optimized | No manual TensorRT-LLM engine building |
| Day-1 deployment | Production-ready in minutes |
| OpenAI-compatible | Drop-in replacement for OpenAI API |
| Auto-scaling | Built-in support for Kubernetes HPA |
| Enterprise support | Part of NVIDIA AI Enterprise |

---

## Architecture

### NIM Components

```
NIM Container
├── API Gateway (OpenAI-compatible endpoints)
├── Request Router (load balancing, batching)
├── Model Runtime
│   ├── TensorRT-LLM Engine (optimized inference)
│   ├── Tokenizer
│   └── KV Cache Manager
├── Triton Backend (model serving)
└── Health & Metrics (Prometheus-compatible)
```

### How NIM Works

1. **Container Pull**: Download pre-built NIM from NGC
2. **Auto-Detection**: NIM detects GPU type and count
3. **Engine Selection**: Chooses optimal TensorRT-LLM engine for hardware
4. **Model Loading**: Loads weights and initializes KV cache
5. **Serving**: Exposes OpenAI-compatible API

### NIM vs Building Your Own

| Aspect | NIM | Custom TensorRT-LLM |
|--------|-----|---------------------|
| Setup time | Minutes | Hours to days |
| Optimization | Automatic | Manual tuning |
| Maintenance | NVIDIA manages | You manage |
| Flexibility | Limited | Full control |
| Cost | License fee | Free (open source) |
| Support | Enterprise support | Community |

---

## Day-1 Deployment

### Prerequisites

```bash
# 1. NGC API Key (get from ngc.nvidia.com)
export NGC_API_KEY="your-api-key"

# 2. Docker with NVIDIA runtime
docker --version
nvidia-smi

# 3. Login to NGC registry
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Quick Start - LLM NIM

```bash
# Pull and run Llama 3.1 8B
docker run -d --gpus all \
  --name llama-nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

# Check health
curl http://localhost:8000/v1/health/ready

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Quick Start - Embedding NIM

```bash
# Run NV-Embed model
docker run -d --gpus all \
  --name embed-nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8001:8000 \
  nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest

# Generate embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-embedqa-e5-v5",
    "input": ["What is machine learning?"],
    "input_type": "query"
  }'
```

### Quick Start - Reranking NIM

```bash
# Run reranker model
docker run -d --gpus all \
  --name rerank-nim \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8002:8000 \
  nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:latest

# Rerank documents
curl -X POST http://localhost:8002/v1/ranking \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-rerankqa-mistral-4b-v3",
    "query": {"text": "What is deep learning?"},
    "passages": [
      {"text": "Deep learning is a subset of machine learning..."},
      {"text": "The weather today is sunny..."}
    ]
  }'
```

---

## Supported Models

### LLM NIMs

| Model | Sizes | Use Case |
|-------|-------|----------|
| Llama 3.1 | 8B, 70B, 405B | General purpose |
| Llama 3.2 | 1B, 3B | Edge/mobile |
| Mistral | 7B | Fast inference |
| Mixtral | 8x7B, 8x22B | MoE, high quality |
| Gemma | 2B, 7B | Lightweight |
| Phi-3 | Mini, Small, Medium | Efficient |
| CodeLlama | 7B, 13B, 34B | Code generation |
| Nemotron | 8B, 70B | NVIDIA optimized |

### Embedding NIMs

| Model | Dimensions | Use Case |
|-------|------------|----------|
| NV-EmbedQA-E5-v5 | 1024 | Enterprise QA |
| NV-Embed-v2 | 4096 | High accuracy |
| E5-large-v2 | 1024 | General purpose |

### Reranking NIMs

| Model | Description |
|-------|-------------|
| NV-RerankQA-Mistral-4B-v3 | High accuracy reranker |
| Rerank-English-v3 | English-focused |

### Multimodal NIMs

| Model | Capabilities |
|-------|--------------|
| VILA | Vision-language |
| NeVA | NVIDIA vision assistant |
| Kosmos | Multimodal understanding |

---

## OpenAI-Compatible API

### Chat Completions

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"  # NIM doesn't require API key for local
)

response = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Text Completions

```python
response = client.completions.create(
    model="meta/llama-3.1-8b-instruct",
    prompt="The three laws of robotics are:",
    max_tokens=200,
    temperature=0.5
)
print(response.choices[0].text)
```

### Embeddings

```python
response = client.embeddings.create(
    model="nvidia/nv-embedqa-e5-v5",
    input=["What is machine learning?", "How do neural networks work?"],
    encoding_format="float"
)

for i, embedding in enumerate(response.data):
    print(f"Embedding {i}: {len(embedding.embedding)} dimensions")
```

### API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat-style generation |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Generate embeddings |
| `/v1/ranking` | POST | Rerank documents |
| `/v1/models` | GET | List available models |
| `/v1/health/ready` | GET | Readiness check |
| `/v1/health/live` | GET | Liveness check |
| `/v1/metrics` | GET | Prometheus metrics |

### Request Parameters

```json
{
  "model": "meta/llama-3.1-8b-instruct",
  "messages": [...],
  "temperature": 0.7,        // 0-2, randomness
  "top_p": 0.9,              // Nucleus sampling
  "top_k": 50,               // Top-k sampling
  "max_tokens": 1024,        // Max output tokens
  "stop": ["\n\n"],          // Stop sequences
  "stream": true,            // Enable streaming
  "presence_penalty": 0.0,   // -2.0 to 2.0
  "frequency_penalty": 0.0,  // -2.0 to 2.0
  "n": 1,                    // Number of completions
  "seed": 42                 // For reproducibility
}
```

---

## Configuration

### Environment Variables

```bash
# Required
NGC_API_KEY=your-key              # NGC authentication

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3      # Specific GPUs
NIM_TENSOR_PARALLEL_SIZE=4        # Tensor parallelism
NIM_PIPELINE_PARALLEL_SIZE=1      # Pipeline parallelism

# Performance Tuning
NIM_MAX_BATCH_SIZE=64             # Max batch size
NIM_MAX_INPUT_LENGTH=4096         # Max input tokens
NIM_MAX_OUTPUT_LENGTH=2048        # Max output tokens
NIM_MAX_MODEL_LEN=8192            # Total context length

# Memory Management
NIM_GPU_MEMORY_UTILIZATION=0.9    # GPU memory fraction
NIM_KV_CACHE_FREE_GPU_MEM_FRACTION=0.8

# Logging
NIM_LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR
NIM_ENABLE_TRACING=true           # OpenTelemetry tracing
```

### Docker Run with Configuration

```bash
docker run -d --gpus '"device=0,1,2,3"' \
  --name llama-nim \
  --shm-size=16g \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_TENSOR_PARALLEL_SIZE=4 \
  -e NIM_MAX_BATCH_SIZE=128 \
  -e NIM_MAX_MODEL_LEN=8192 \
  -e NIM_GPU_MEMORY_UTILIZATION=0.9 \
  -e NIM_LOG_LEVEL=INFO \
  -p 8000:8000 \
  -v /path/to/cache:/opt/nim/.cache \
  nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
```

### Multi-GPU Configuration

```bash
# 70B model with 4-way tensor parallelism
docker run -d --gpus '"device=0,1,2,3"' \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_TENSOR_PARALLEL_SIZE=4 \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-70b-instruct:latest

# 405B model with 8-way tensor parallelism
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_TENSOR_PARALLEL_SIZE=8 \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-405b-instruct:latest
```

---

## Kubernetes Deployment

### Basic Deployment

```yaml
# nim-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-nim
  labels:
    app: llama-nim
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama-nim
  template:
    metadata:
      labels:
        app: llama-nim
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
        ports:
        - containerPort: 8000
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ngc-secret
              key: api-key
        - name: NIM_MAX_BATCH_SIZE
          value: "64"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        livenessProbe:
          httpGet:
            path: /v1/health/live
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /v1/health/ready
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 5
        volumeMounts:
        - name: cache
          mountPath: /opt/nim/.cache
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: nim-cache-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"
      imagePullSecrets:
      - name: ngc-registry-secret
```

### Service and Ingress

```yaml
# nim-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-nim-service
spec:
  selector:
    app: llama-nim
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llama-nim-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
spec:
  rules:
  - host: llm-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llama-nim-service
            port:
              number: 8000
```

### Horizontal Pod Autoscaler

```yaml
# nim-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-nim-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-nim
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Pods
    pods:
      metric:
        name: nim_request_queue_size
      target:
        type: AverageValue
        averageValue: "5"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
```

### NGC Secret Setup

```bash
# Create NGC registry secret
kubectl create secret docker-registry ngc-registry-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY

# Create API key secret
kubectl create secret generic ngc-secret \
  --from-literal=api-key=$NGC_API_KEY
```

---

## Custom Model Deployment

### Using LoRA Adapters with NIM

```bash
# Mount your LoRA adapter
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_PEFT_SOURCE=/models/my-lora-adapter \
  -v /path/to/lora:/models/my-lora-adapter \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Custom Fine-Tuned Models

```bash
# For NeMo-trained models
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_NAME=my-custom-model \
  -v /path/to/nemo/model:/models/custom \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemo-llm-nim:latest
```

### Model Profile Selection

```bash
# List available profiles for hardware
docker run --rm --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest \
  list-profiles

# Use specific profile
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_PROFILE=throughput-optimized \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

---

## Monitoring & Observability

### Prometheus Metrics

NIM exposes metrics at `/v1/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nim'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /v1/metrics
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `nim_request_total` | Total requests |
| `nim_request_duration_seconds` | Request latency |
| `nim_tokens_generated_total` | Total output tokens |
| `nim_tokens_per_second` | Generation throughput |
| `nim_queue_size` | Pending requests |
| `nim_batch_size` | Current batch size |
| `nim_gpu_memory_used_bytes` | GPU memory usage |
| `nim_kv_cache_utilization` | KV cache usage |

### Grafana Dashboard

```json
{
  "panels": [
    {
      "title": "Request Latency (p99)",
      "targets": [{
        "expr": "histogram_quantile(0.99, sum(rate(nim_request_duration_seconds_bucket[5m])) by (le))"
      }]
    },
    {
      "title": "Tokens per Second",
      "targets": [{
        "expr": "rate(nim_tokens_generated_total[1m])"
      }]
    },
    {
      "title": "GPU Memory Usage",
      "targets": [{
        "expr": "nim_gpu_memory_used_bytes / nim_gpu_memory_total_bytes * 100"
      }]
    },
    {
      "title": "Queue Depth",
      "targets": [{
        "expr": "nim_queue_size"
      }]
    }
  ]
}
```

### OpenTelemetry Tracing

```bash
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_ENABLE_TRACING=true \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

---

## Performance Tuning

### Throughput Optimization

```bash
# Maximize throughput
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MAX_BATCH_SIZE=256 \
  -e NIM_MAX_NUM_SEQS=256 \
  -e NIM_GPU_MEMORY_UTILIZATION=0.95 \
  -e NIM_ENABLE_CHUNKED_PREFILL=true \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Latency Optimization

```bash
# Minimize latency
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MAX_BATCH_SIZE=8 \
  -e NIM_MAX_NUM_SEQS=8 \
  -e NIM_SCHEDULING_POLICY=fcfs \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Memory Optimization

```bash
# For memory-constrained environments
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_GPU_MEMORY_UTILIZATION=0.7 \
  -e NIM_MAX_MODEL_LEN=4096 \
  -e NIM_QUANTIZATION=fp8 \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Performance Benchmarking

```python
import time
import concurrent.futures
import requests

def benchmark_nim(url, num_requests=100, concurrency=10):
    """Benchmark NIM throughput and latency."""

    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }

    latencies = []

    def make_request():
        start = time.time()
        response = requests.post(f"{url}/v1/chat/completions", json=payload)
        latency = time.time() - start
        return latency, response.status_code

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            latency, status = future.result()
            if status == 200:
                latencies.append(latency)

    total_time = time.time() - start_time

    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(latencies)}")
    print(f"Throughput: {len(latencies)/total_time:.2f} req/s")
    print(f"Avg latency: {sum(latencies)/len(latencies)*1000:.2f} ms")
    print(f"P50 latency: {sorted(latencies)[len(latencies)//2]*1000:.2f} ms")
    print(f"P99 latency: {sorted(latencies)[int(len(latencies)*0.99)]*1000:.2f} ms")

benchmark_nim("http://localhost:8000", num_requests=100, concurrency=10)
```

---

## NIM vs TensorRT-LLM Comparison

| Aspect | NIM | TensorRT-LLM Direct |
|--------|-----|---------------------|
| **Setup Time** | Minutes | Hours/Days |
| **Optimization** | Automatic | Manual engine building |
| **API** | OpenAI-compatible | Custom or Triton |
| **Updates** | Container pull | Rebuild engine |
| **Customization** | Limited | Full control |
| **Multi-GPU** | Automatic TP | Manual configuration |
| **Cost** | License required | Free |
| **Support** | Enterprise | Community |
| **Latency** | Near-optimal | Optimal (if tuned) |
| **Throughput** | Near-optimal | Optimal (if tuned) |

### When to Use NIM

- Quick deployment without optimization expertise
- Standard model architectures
- Enterprise support requirements
- OpenAI API compatibility needed

### When to Use TensorRT-LLM Directly

- Maximum performance required
- Custom model architectures
- Cost-sensitive deployments
- Full control over optimization

---

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker logs llama-nim

# Common causes:
# - Invalid NGC_API_KEY
# - Insufficient GPU memory
# - Wrong GPU count for model size
```

**Out of memory:**
```bash
# Reduce memory usage
-e NIM_GPU_MEMORY_UTILIZATION=0.7
-e NIM_MAX_MODEL_LEN=2048
-e NIM_MAX_BATCH_SIZE=16
```

**Slow startup:**
```bash
# Use persistent cache
-v /path/to/cache:/opt/nim/.cache

# First run downloads and optimizes model
# Subsequent runs use cached engine
```

**Connection refused:**
```bash
# Wait for model to load (can take 2-5 minutes)
# Check readiness endpoint
curl http://localhost:8000/v1/health/ready
```

### Debug Mode

```bash
docker run -it --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_LOG_LEVEL=DEBUG \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

---

## Best Practices

1. **Use persistent cache** - Mount volume to `/opt/nim/.cache` for faster restarts
2. **Set resource limits** - Prevent OOM with proper memory/GPU limits
3. **Enable health checks** - Use liveness/readiness probes in K8s
4. **Monitor metrics** - Set up Prometheus/Grafana dashboards
5. **Use appropriate model size** - Match model to available GPU memory
6. **Configure batching** - Tune batch size for your latency/throughput needs
7. **Implement retries** - Handle transient failures gracefully
8. **Secure endpoints** - Use API gateway for authentication in production
9. **Plan for scaling** - Use HPA with queue-based metrics
10. **Keep NIMs updated** - Pull latest containers for improvements

---

## Resources

- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NGC Catalog - NIM](https://catalog.ngc.nvidia.com/nim)
- [NIM API Reference](https://docs.nvidia.com/nim/api/)
- [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)
