---
name: llm-quiz
description: Generate quiz questions and practice problems for LLM topics. Use when testing knowledge, creating flashcards, doing active recall practice, or preparing for exams on large language models, NVIDIA tools, or ML infrastructure.
allowed-tools: Read, Grep, Glob, Edit
---

# LLM Quiz Generator

Generate quizzes using Active Recall and Interleaved Practice techniques.

## Instructions

### 1. Quiz Configuration

Ask the user:
- **Topics**: Which areas? (Infrastructure, Data-Training, Architecture, Operations, Fundamentals)
- **Difficulty**: Easy, Medium, Hard, or Mixed
- **Question Count**: How many questions?
- **Format**: Multiple choice, fill-in-blank, short answer, or mixed

### 2. Question Types

**Multiple Choice**
```
Q: Which quantization method migrates difficulty from activations to weights?
A) FP8
B) INT8 SmoothQuant  ✓
C) INT4 AWQ
D) W4A8
```

**Fill-in-the-Blank**
```
Q: _______ parallelism splits model layers across GPUs, where each GPU holds a slice of weight matrices.
A: Tensor Parallel (TP)
```

**Short Answer**
```
Q: Explain the difference between PTQ and QAT.
A: PTQ (Post-Training Quantization) is faster and doesn't require retraining. QAT (Quantization-Aware Training) provides better accuracy but requires training.
```

**True/False**
```
Q: NVLink provides higher bandwidth than PCIe 5.0 for GPU-to-GPU communication.
A: True (NVLink: 900 GB/s on H100 vs PCIe 5.0: 64 GB/s)
```

**Matching**
```
Match the parallelism strategy with its description:
1. Tensor Parallel    → Splits weight matrices across GPUs
2. Pipeline Parallel  → Splits model by layers
3. Data Parallel      → Each GPU holds full model copy
4. Sequence Parallel  → Splits sequence dimension
```

### 3. Topic Coverage by Weight

Generate questions proportionally based on exam weights:

| Topic | Weight | Source File | Key Areas |
|-------|--------|-------------|-----------|
| Model Optimization | 17% | `1 - infrastructure/1_1-model-optimization.md` | Quantization, TensorRT-LLM, AMAX, KV-cache |
| GPU Acceleration | 14% | `1 - infrastructure/1_2-gpu-acceleration.md` | Parallelism, NVLink, NCCL, Tensor Cores |
| Prompt Engineering | 13% | `3 - architecture/3_1-prompt-engineering.md` | Zero/few-shot, CoT, temperature, top_p |
| Fine-Tuning | 13% | `2 - data-training/2_1-fine-tuning.md` | LoRA, QLoRA, RLHF, DPO, SFT |
| Deployment | 9% | `1 - infrastructure/1_3-deployment.md` | Triton, config.pbtxt, dynamic batching |
| Data Preparation | 9% | `2 - data-training/2_2-data-preparation.md` | Tokenization, BPE, RAPIDS |
| Evaluation | 7% | `4 - operations/4_1-evaluation.md` | Perplexity, BLEU, ROUGE |
| Monitoring | 7% | `4 - operations/4_2-monitoring.md` | Latency, throughput, GPU utilization |
| LLM Architecture | 6% | `3 - architecture/3_2-llm-architecture.md` | Transformers, attention, positional encoding |
| Safety | 5% | `4 - operations/4_3-safety.md` | NeMo Guardrails, jailbreak detection |

### 4. Interleaved Practice Mode

When enabled:
- Mix questions from different topics randomly
- Don't group by category
- This improves long-term retention

### 5. Feedback Protocol

After each answer:
1. Indicate correct/incorrect
2. Provide brief explanation
3. Reference the relevant knowledge base section
4. Track score

### 6. End of Quiz

Provide:
- Final score (X/Y correct, percentage)
- Breakdown by topic
- Weak areas to review
- Recommended next steps

### 7. Logging

After each quiz session, log to `logs/learning.log`:
```
[YYYY-MM-DD HH:MM:SS] [QUIZ] [SESSION] Topics: <topics>, Score: X/Y (<percentage>%), Weak: <weak_areas>
```
