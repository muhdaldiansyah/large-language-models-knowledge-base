# 4.2 Production Monitoring (7%)

## Overview

Production monitoring ensures LLM systems are performant, reliable, and meeting SLAs.

```
LLM Service → Metrics Collection → Dashboards → Alerts → Response
                    ↓
              Logging & Tracing
```

---

## Key Performance Metrics

### Latency Metrics

**Time to First Token (TTFT)**
- Time from request to first generated token
- Critical for interactive applications
- Target: < 500ms for chat applications

**Inter-Token Latency (ITL)**
- Time between consecutive tokens
- Affects perceived streaming speed
- Target: < 50ms for smooth streaming

**End-to-End Latency**
- Total time from request to complete response
- Depends on output length
- Formula: TTFT + (num_tokens × ITL)

**Percentile Tracking**
```
p50 (median): Typical user experience
p95: Most users (exclude outliers)
p99: Worst-case (important for SLAs)
p99.9: Tail latency (capacity planning)
```

```python
import numpy as np

latencies = [...]  # collected latencies

metrics = {
    'p50': np.percentile(latencies, 50),
    'p95': np.percentile(latencies, 95),
    'p99': np.percentile(latencies, 99),
    'mean': np.mean(latencies),
    'std': np.std(latencies)
}
```

### Throughput Metrics

**Tokens per Second (TPS)**
```
TPS = Total output tokens / Total time
```

**Requests per Second (RPS)**
```
RPS = Total requests / Total time
```

**Concurrent Requests**
- Number of simultaneous in-flight requests
- Limited by GPU memory and batch size

**Queue Depth**
- Requests waiting to be processed
- High queue = capacity issue

### Resource Utilization

**GPU Utilization**
```bash
# NVIDIA SMI
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1

# Programmatic
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU: {util.gpu}%, Memory: {util.memory}%")
```

**GPU Memory**
```
Used memory / Total memory

Components:
- Model weights
- KV cache
- Activations
- Framework overhead
```

**CPU Utilization**
- Preprocessing/postprocessing
- Tokenization
- Data loading

**Network I/O**
- Request/response sizes
- Bandwidth utilization
- For distributed inference: inter-GPU communication

### Error Metrics

**Error Rate**
```
Error rate = Failed requests / Total requests × 100%
```

**Error Types:**
| Error | Cause | Action |
|-------|-------|--------|
| Timeout | Slow inference | Scale up / optimize |
| OOM | Insufficient memory | Reduce batch size |
| Rate limit | Too many requests | Queue / scale out |
| Model error | Internal failure | Debug / restart |
| Input validation | Bad request | Return 400 |

**Success Rate / Availability**
```
Availability = Successful requests / Total requests × 100%

Target: 99.9% = ~8.7 hours downtime/year
        99.99% = ~52 minutes downtime/year
```

---

## Logging Best Practices

### What to Log

```python
log_entry = {
    # Request identification
    "request_id": "uuid-xxx",
    "timestamp": "2024-01-15T10:30:00Z",
    "user_id": "user-123",

    # Input
    "prompt_length": 150,
    "prompt_hash": "sha256:xxx",  # For dedup, not PII

    # Output
    "output_length": 500,
    "finish_reason": "stop",  # stop, length, error

    # Performance
    "ttft_ms": 120,
    "total_latency_ms": 2500,
    "tokens_per_second": 45,

    # Resources
    "gpu_utilization": 85,
    "memory_used_gb": 24,

    # Model info
    "model_name": "llama-3.1-8b",
    "model_version": "v1.2",

    # Status
    "status": "success",
    "error": None
}
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

def inference_with_logging(request):
    logger.info("request_started",
                request_id=request.id,
                prompt_length=len(request.prompt))

    try:
        start = time.time()
        response = model.generate(request.prompt)
        latency = time.time() - start

        logger.info("request_completed",
                    request_id=request.id,
                    latency_ms=latency * 1000,
                    output_tokens=len(response.tokens))

        return response

    except Exception as e:
        logger.error("request_failed",
                     request_id=request.id,
                     error=str(e))
        raise
```

### Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed debugging info |
| INFO | Normal operations |
| WARNING | Recoverable issues |
| ERROR | Failed requests |
| CRITICAL | System failures |

### Privacy Considerations

- **DO NOT log:** Full prompts, user data, PII
- **DO log:** Hashed/truncated prompts, metadata, metrics
- **Consider:** Data retention policies, GDPR compliance

---

## Distributed Tracing

Track requests across multiple services.

```
Client → API Gateway → Load Balancer → Inference Server → Model
   ↓          ↓              ↓               ↓              ↓
 span1      span2          span3          span4          span5
   └──────────────────────────────────────────────────────────┘
                         Trace ID: xyz-123
```

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Usage
@tracer.start_as_current_span("inference")
def run_inference(prompt):
    span = trace.get_current_span()
    span.set_attribute("prompt.length", len(prompt))

    with tracer.start_as_current_span("tokenize"):
        tokens = tokenizer.encode(prompt)

    with tracer.start_as_current_span("generate"):
        output = model.generate(tokens)

    span.set_attribute("output.length", len(output))
    return output
```

---

## Health Checks

### Liveness Probe

"Is the service running?"

```python
@app.get("/health/live")
def liveness():
    return {"status": "alive"}
```

### Readiness Probe

"Is the service ready to accept traffic?"

```python
@app.get("/health/ready")
def readiness():
    # Check model is loaded
    if not model.is_loaded():
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "model loading"}
        )

    # Check GPU memory
    if get_gpu_memory_free() < MIN_FREE_MEMORY:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "low memory"}
        )

    return {"status": "ready"}
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: llm-server
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 5
```

---

## Alerting

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| p99 Latency | > 5s | > 10s |
| Error Rate | > 1% | > 5% |
| GPU Util | > 90% | > 95% |
| Memory | > 80% | > 90% |
| Queue Depth | > 100 | > 500 |

### Alert Configuration (Prometheus)

```yaml
groups:
- name: llm_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, llm_latency_bucket) > 10
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High p99 latency detected"

  - alert: HighErrorRate
    expr: rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Error rate above 5%"

  - alert: GPUMemoryHigh
    expr: gpu_memory_used / gpu_memory_total > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage above 90%"
```

---

## Model Versioning

### Version Tracking

```python
model_metadata = {
    "name": "llama-3.1-8b-instruct",
    "version": "v1.2.0",
    "checksum": "sha256:abc123...",
    "trained_date": "2024-01-10",
    "config": {
        "quantization": "fp8",
        "max_batch_size": 32,
        "tensor_parallel": 2
    },
    "metrics": {
        "mmlu_accuracy": 0.72,
        "latency_p99": 2.5
    }
}
```

### Canary Deployments

```
Traffic → 95% → Production Model (v1.1)
       → 5%  → Canary Model (v1.2)
                    ↓
              Monitor metrics
                    ↓
         If good: gradual rollout
         If bad: rollback
```

### A/B Testing

```python
def route_request(request, user_id):
    # Consistent routing based on user
    bucket = hash(user_id) % 100

    if bucket < 10:  # 10% to new model
        return model_v2.generate(request)
    else:
        return model_v1.generate(request)
```

---

## Retraining Triggers

### When to Retrain

| Signal | Indicator | Action |
|--------|-----------|--------|
| Performance drift | Metrics declining | Evaluate & retrain |
| Data drift | Input distribution changed | Update training data |
| Concept drift | Task requirements changed | Re-evaluate objectives |
| New data | Fresh labeled data | Incremental training |
| Model update | Better base model available | Fine-tune new base |

### Monitoring for Drift

```python
from scipy import stats

def detect_drift(reference_dist, current_dist, threshold=0.05):
    """KS test for distribution drift"""
    statistic, p_value = stats.ks_2samp(reference_dist, current_dist)
    return p_value < threshold  # True if drift detected

# Monitor embedding distributions
reference_embeddings = get_reference_embeddings()
current_embeddings = get_recent_embeddings()

if detect_drift(reference_embeddings, current_embeddings):
    alert("Data drift detected - consider retraining")
```

---

## NVIDIA Monitoring Tools

### Nsight Systems

Profile GPU workloads:
```bash
nsys profile -o profile_output python inference.py
nsys stats profile_output.nsys-rep
```

**Captures:**
- CUDA API calls
- GPU kernels
- Memory transfers
- CPU/GPU timeline

### DCGM (Data Center GPU Manager)

Enterprise GPU monitoring:
```bash
# Start DCGM
dcgmi discovery -l

# Monitor metrics
dcgmi dmon -e 100,101,102,103

# Health check
dcgmi health -c
```

**Metrics:**
- Temperature, power
- Utilization, memory
- ECC errors
- NVLink bandwidth

### Triton Metrics

Built-in Prometheus metrics:
```
# Request metrics
nv_inference_request_success
nv_inference_request_failure
nv_inference_request_duration_us

# Queue metrics
nv_inference_queue_duration_us
nv_inference_pending_request_count

# GPU metrics
nv_gpu_utilization
nv_gpu_memory_used_bytes
```

---

## Monitoring Stack Example

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC

  llm-server:
    build: .
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    ports:
      - "8000:8000"
```

---

## Dashboard Essentials

### Key Panels

1. **Request Rate:** RPS over time
2. **Latency Distribution:** p50/p95/p99 histogram
3. **Error Rate:** % errors with breakdown
4. **GPU Utilization:** Per-GPU utilization
5. **Memory Usage:** GPU memory over time
6. **Queue Depth:** Pending requests
7. **Token Throughput:** Tokens/second
8. **Model Comparison:** A/B test metrics

### Grafana Query Examples

```promql
# Request rate
rate(llm_requests_total[5m])

# p99 latency
histogram_quantile(0.99, rate(llm_latency_seconds_bucket[5m]))

# Error rate percentage
100 * rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])

# GPU utilization average
avg(gpu_utilization_percent)

# Tokens per second
rate(llm_output_tokens_total[1m])
```

---

## Best Practices

1. **Monitor what matters:** Focus on user-facing metrics
2. **Set meaningful SLOs:** Based on user requirements
3. **Alert on symptoms:** Not just causes
4. **Use percentiles:** Mean hides outliers
5. **Correlate metrics:** GPU util + latency + errors
6. **Automate responses:** Auto-scale on queue depth
7. **Retain history:** For trend analysis and debugging
8. **Test alerts:** Ensure they fire when expected
