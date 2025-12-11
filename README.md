# Large Language Models Knowledge Base

A comprehensive reference for production LLM systems using the NVIDIA ecosystem.

## Repository Structure

```
├── 1 - infrastructure/        # Model optimization, GPU, deployment (40%)
├── 2 - data-training/         # Fine-tuning, data preparation (22%)
├── 3 - architecture/          # Prompt engineering, transformers, RAG (19%)
├── 4 - operations/            # Evaluation, monitoring, safety (19%)
├── 5 - fundamentals/          # ML basics, NeMo 2.0, infrastructure
├── logs/                      # Learning activity tracking
├── .claude/skills/            # 5 custom learning skills
└── learn.md                   # 22 learning techniques
```

## Topics

### 1 - Infrastructure (40% weight)
| File | Topic | Weight |
|------|-------|--------|
| `1_1-model-optimization.md` | Quantization (FP8, INT8, INT4), TensorRT-LLM, mixed precision | 17% |
| `1_2-gpu-acceleration.md` | Tensor/Pipeline/Data parallelism, NVLink, NCCL | 14% |
| `1_3-deployment.md` | Triton Inference Server, Kubernetes, dynamic batching | 9% |
| `1_4-nvidia-nim.md` | Day-1 deployment, OpenAI-compatible API | - |

### 2 - Data & Training (22% weight)
| File | Topic | Weight |
|------|-------|--------|
| `2_1-fine-tuning.md` | LoRA, QLoRA, P-Tuning, SFT, RLHF, DPO | 13% |
| `2_2-data-preparation.md` | Tokenization (BPE, WordPiece), RAPIDS | 9% |
| `2_3-data-curation.md` | Deduplication, quality filtering, PII redaction | - |

### 3 - Architecture & Applications (19% weight)
| File | Topic | Weight |
|------|-------|--------|
| `3_1-prompt-engineering.md` | Zero/few-shot, CoT, temperature, structured outputs | 13% |
| `3_2-llm-architecture.md` | Transformers, attention (MHA/MQA/GQA), RoPE, FlashAttention | 6% |
| `3_3-rag.md` | Embeddings, vector DBs, chunking, reranking, function calling | - |

### 4 - Operations (19% weight)
| File | Topic | Weight |
|------|-------|--------|
| `4_1-evaluation.md` | Perplexity, BLEU, ROUGE, benchmarks, LLM-as-judge | 7% |
| `4_2-monitoring.md` | Latency metrics, throughput, logging, tracing | 7% |
| `4_3-safety.md` | NeMo Guardrails, Colang, jailbreak detection, compliance | 5% |

### 5 - Fundamentals
| File | Topic |
|------|-------|
| `5_1-ml-fundamentals.md` | Neural networks, backprop, NER, t-SNE, activation functions |
| `5_2-nemo-2.md` | NeMo 2.0, NeMo-Run, Slurm, Kubernetes, DGX SuperPOD |

## Learning Tools

### Custom Skills (`.claude/skills/`)
- **llm-tutor** - Comprehensive teaching with learning techniques
- **llm-quiz** - Generate practice questions and quizzes
- **llm-explainer** - Dual-level explanations (beginner + advanced)
- **socratic-llm** - Socratic questioning method
- **exam-prep** - Weighted study plans and exam preparation

### Learning Methodology
See `learn.md` for 22 techniques including:
- Scaffolded Learning, Dual Coding, Worked Examples
- Socratic Method, Feynman Technique
- Spaced Repetition, Active Recall, Interleaved Practice

## Key Technologies

| Category | Technologies |
|----------|--------------|
| NVIDIA Stack | NeMo, TensorRT-LLM, Triton, NIM, NCCL, Nsight |
| Quantization | FP8, INT8, INT4, AWQ, GPTQ, SmoothQuant |
| Parallelism | Tensor, Pipeline, Data, Sequence |
| Fine-tuning | LoRA, QLoRA, RLHF, DPO, SFT |
| RAG | Milvus, FAISS, Pinecone, NV-Embed |

## Resources

- [NVIDIA NeMo](https://developer.nvidia.com/nemo)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Inference Server](https://github.com/triton-inference-server)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
