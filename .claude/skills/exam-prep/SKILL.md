---
name: exam-prep
description: Prepare for LLM certification exams with focused study plans and practice. Use when preparing for NVIDIA certification, studying for exams, creating study schedules, or doing targeted review of weighted topics on large language models.
allowed-tools: Read, Grep, Glob, Edit
---

# LLM Exam Preparation

Structured exam preparation focusing on weighted topics and strategic study.

## Instructions

### 1. Exam Topic Weights

Prioritize study based on these weights:

| Priority | Topic | Weight | Source File | Key Areas |
|----------|-------|--------|-------------|-----------|
| 1 | Model Optimization | 17% | `1 - infrastructure/1_1-model-optimization.md` | FP8, INT8, INT4, AMAX, KV-cache, PagedAttention, PTQ/QAT |
| 2 | GPU Acceleration | 14% | `1 - infrastructure/1_2-gpu-acceleration.md` | TP, PP, DP, SP, NVLink, NCCL, Tensor Cores, Nsight |
| 3 | Prompt Engineering | 13% | `3 - architecture/3_1-prompt-engineering.md` | Zero/few-shot, CoT, temperature, top_p, top_k |
| 4 | Fine-Tuning | 13% | `2 - data-training/2_1-fine-tuning.md` | LoRA, QLoRA, P-Tuning, SFT, RLHF, DPO |
| 5 | Deployment | 9% | `1 - infrastructure/1_3-deployment.md` | Triton, config.pbtxt, dynamic_batching, Kubernetes |
| 6 | Data Preparation | 9% | `2 - data-training/2_2-data-preparation.md` | BPE, WordPiece, SentencePiece, RAPIDS |
| 7 | Evaluation | 7% | `4 - operations/4_1-evaluation.md` | Perplexity, BLEU, ROUGE, F1 |
| 8 | Monitoring | 7% | `4 - operations/4_2-monitoring.md` | Latency p50/p95/p99, throughput, GPU utilization |
| 9 | LLM Architecture | 6% | `3 - architecture/3_2-llm-architecture.md` | Attention, FFN, LayerNorm, RoPE, FlashAttention |
| 10 | Safety | 5% | `4 - operations/4_3-safety.md` | NeMo Guardrails, Colang, jailbreak detection |

### 2. Study Plan Generator

Based on available time, create focused plans:

**1-Day Cram (8 hours)**
- 2h: Model Optimization (17%)
- 1.5h: GPU Acceleration (14%)
- 1.5h: Prompt Engineering + Fine-Tuning (26%)
- 1h: Deployment + Data Prep (18%)
- 1h: Evaluation + Monitoring (14%)
- 1h: Architecture + Safety (11%)

**1-Week Plan**
- Day 1-2: Model Optimization deep dive
- Day 3: GPU Acceleration & Parallelism
- Day 4: Prompt Engineering + Fine-Tuning methods
- Day 5: Deployment (Triton) + Data Preparation
- Day 6: Evaluation, Monitoring, Safety
- Day 7: Full review + practice exams

**2-Week Plan**
- Week 1: Learn all topics with examples
- Week 2: Practice problems + weak area review

### 3. Must-Know Facts

**Model Optimization**
- FP8 requires Hopper architecture (H100)
- SmoothQuant migrates quantization difficulty from activations to weights
- AWQ = Activation-aware Weight Quantization
- GPTQ uses approximate second-order information
- PagedAttention eliminates memory fragmentation

**GPU Acceleration**
- NVLink: 900 GB/s (H100), PCIe 5.0: 64 GB/s
- Tensor Parallel: splits weight matrices (needs NVLink)
- Pipeline Parallel: splits by layers (introduces bubbles)
- Data Parallel: full model per GPU, batch split

**Triton Server**
- Model repo structure: `model_name/config.pbtxt` + `1/model.plan`
- Key params: max_batch_size, instance_group, dynamic_batching
- max_queue_delay_microseconds controls latency vs throughput

**Fine-Tuning**
- LoRA: Low-Rank Adaptation (efficient, few trainable params)
- QLoRA: Quantized LoRA (4-bit base model)
- RLHF: Reinforcement Learning from Human Feedback
- DPO: Direct Preference Optimization (simpler than RLHF)

**NeMo Guardrails**
- Config: config.yml + Colang (.co files)
- 5 rail types: input, dialog, output, retrieval, execution

### 4. Common Exam Question Patterns

**"Which is best for..."**
- Memory efficiency → Quantization, LoRA
- Multi-GPU large model → Tensor Parallel
- Very deep model → Pipeline Parallel
- Scale training batch → Data Parallel
- Fast deployment → NVIDIA NIM

**"What is the purpose of..."**
- AMAX calibration → Determine quantization scaling factors
- KV-cache → Store key/value vectors to avoid recomputation
- In-flight batching → Dynamic request grouping during inference
- Colang → Define conversational guardrails

**Calculate/Estimate**
- FLOPs forward: ~2 * Parameters * Tokens
- FLOPs training: ~6 * Parameters * Tokens
- Model memory: Parameters × Bytes per parameter

### 5. Spaced Repetition Schedule

After initial study:
- Day 1: Review all
- Day 3: Review weak areas
- Day 7: Full review
- Day 14: Practice exam
- Day 28: Final review

### 6. Practice Exam Mode

Generate realistic exam conditions:
1. Timed sections
2. No reference materials
3. Mixed question types
4. Score tracking
5. Detailed review after

### 7. Weak Area Protocol

When user struggles with a topic:
1. Identify specific gap
2. Provide targeted mini-lesson
3. Generate 3-5 focused practice questions
4. Re-assess understanding
5. Schedule for spaced review

### 8. Pre-Exam Checklist

- [ ] Can explain all quantization methods and when to use each
- [ ] Know parallelism strategies and hardware requirements
- [ ] Understand Triton config.pbtxt parameters
- [ ] Can write TensorRT-LLM build commands
- [ ] Know fine-tuning methods and trade-offs
- [ ] Understand tokenization algorithms
- [ ] Can calculate FLOPs and memory requirements
- [ ] Know evaluation metrics and when to use each
- [ ] Understand NeMo Guardrails configuration
- [ ] Familiar with NVIDIA tool ecosystem

### 9. Logging

After exam prep sessions, log to `logs/learning.log`:
```
[YYYY-MM-DD HH:MM:SS] [REVIEW] [EXAM-PREP] Plan: <plan_type>, Topics: <topics_covered>, Progress: <percentage>%
```
