# Group D - Operations

## D1. Evaluation (7%)

### Metrics
- Perplexity
- BLEU
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- Accuracy / F1
- Exact Match
- BERTScore

### Techniques
- Cross-validation
- Train/Val/Test split
- A/B testing
- Error analysis
- Benchmarking (MMLU, HellaSwag, GSM8K)

### LLM-as-Judge
- Evaluation prompts
- Rubric design
- Position bias mitigation

**Detailed notes:** [4_1-evaluation.md](4_1-evaluation.md)

---

## D2. Production Monitoring (7%)

### Performance Metrics
- Latency (TTFT, ITL, p50/p95/p99)
- Throughput (tokens/sec, RPS)
- GPU utilization
- Memory usage
- Error rates

### Operations
- Structured logging
- Distributed tracing (OpenTelemetry)
- Health checks (liveness/readiness)
- Alerting and SLOs

### Model Lifecycle
- Version tracking
- Canary deployments
- A/B testing
- Retraining triggers

### NVIDIA Tools
- Nsight Systems
- DCGM (Data Center GPU Manager)
- Triton metrics

**Detailed notes:** [4_2-monitoring.md](4_2-monitoring.md)

---

## D3. Safety, Ethics & Compliance (5%)

### NeMo Guardrails
- config.yml configuration
- Colang 2.0 (.co files)

### Five Types of Rails
1. Input rails
2. Dialog rails
3. Output rails
4. Retrieval rails
5. Execution rails

### Built-in Safety Features
- Jailbreak detection
- Self-check input/output
- Fact checking
- Hallucination detection
- PII masking
- Llama Guard moderation

### Bias & Fairness
- Bias types (selection, measurement, algorithmic)
- Bias detection methods
- Mitigation strategies
- Fairness metrics

### Compliance
- GDPR considerations
- AI Act (EU) risk levels
- Industry-specific regulations

**Detailed notes:** [4_3-safety.md](4_3-safety.md)

---

## D4. NVIDIA Tools Reference

| Tool | Purpose |
|------|---------|
| NeMo | Training framework |
| TensorRT-LLM | Inference optimization |
| Triton | Inference server |
| Nsight Systems | GPU profiling |
| NeMo Guardrails | Safety rails |
| NGC Catalog | Model/container registry |
| NCCL | Multi-GPU communication |
| ModelOpt | Model optimization |
