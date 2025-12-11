# LLM Knowledge Base - Project Context

## Project Overview

This is a comprehensive knowledge base for learning about Large Language Models (LLMs) with focus on the NVIDIA ecosystem and production deployment.

## Repository Structure

```
├── 1 - infrastructure/             # Infrastructure topics (40% weight)
│   ├── 1_1-model-optimization.md      # Quantization, Pruning, Distillation, Speculative Decoding (17%)
│   ├── 1_2-gpu-acceleration.md        # Parallelism, FlashAttention, NCCL, Memory Optimization (14%)
│   ├── 1_3-deployment.md              # Triton, vLLM, Continuous Batching, ONNX (9%)
│   ├── 1_4-nvidia-nim.md              # NIM deployment
│   ├── 1_5-nvidia-ecosystem.md        # TAO, NGC, Base Command, AI Enterprise
│   └── README.md
├── 2 - data-training/              # Data & Training topics (22% weight)
│   ├── 2_1-fine-tuning.md             # LoRA, QLoRA, RLHF, DPO (13%)
│   ├── 2_2-data-preparation.md        # Tokenization, preprocessing (9%)
│   ├── 2_3-data-curation.md           # Quality filtering, deduplication
│   └── README.md
├── 3 - architecture/               # Architecture topics (19% weight)
│   ├── 3_1-prompt-engineering.md      # Prompting techniques (13%)
│   ├── 3_2-llm-architecture.md        # Transformers, attention (6%)
│   ├── 3_3-rag.md                     # RAG pipeline, vector DBs
│   └── README.md
├── 4 - operations/                 # Operations topics (19% weight)
│   ├── 4_1-evaluation.md              # Metrics, benchmarks (7%)
│   ├── 4_2-monitoring.md              # Production monitoring (7%)
│   ├── 4_3-safety.md                  # Guardrails, compliance (5%)
│   └── README.md
├── 5 - fundamentals/               # Fundamentals
│   ├── 5_1-ml-fundamentals.md         # Neural networks, NER, t-SNE
│   ├── 5_2-nemo-2.md                  # NeMo 2.0, Slurm, K8s, DGX
│   └── README.md
├── logs/                           # Learning activity logs
│   ├── learning.log                   # Activity tracking
│   └── README.md
├── .claude/skills/                 # Custom Claude skills
│   ├── llm-tutor/
│   ├── llm-quiz/
│   ├── llm-explainer/
│   ├── socratic-llm/
│   └── exam-prep/
├── learn.md                        # 22 learning techniques methodology
├── CLAUDE.md                       # This file - project instructions
└── README.md                       # Main readme
```

## Content Purpose

- Study material for NVIDIA LLM certification/exams
- Reference guide for production LLM systems
- Learning resource with weighted topic importance

## Topic Weights (for exam focus)

| Topic | Weight | File |
|-------|--------|------|
| Model Optimization | 17% | `1 - infrastructure/1_1-model-optimization.md` |
| GPU Acceleration | 14% | `1 - infrastructure/1_2-gpu-acceleration.md` |
| Prompt Engineering | 13% | `3 - architecture/3_1-prompt-engineering.md` |
| Fine-Tuning | 13% | `2 - data-training/2_1-fine-tuning.md` |
| Deployment | 9% | `1 - infrastructure/1_3-deployment.md` |
| Data Preparation | 9% | `2 - data-training/2_2-data-preparation.md` |
| Evaluation | 7% | `4 - operations/4_1-evaluation.md` |
| Monitoring | 7% | `4 - operations/4_2-monitoring.md` |
| LLM Architecture | 6% | `3 - architecture/3_2-llm-architecture.md` |
| Safety | 5% | `4 - operations/4_3-safety.md` |

## Available Skills

This project has 5 custom skills in `.claude/skills/`:

- **llm-tutor**: Comprehensive teaching with learning techniques
- **llm-quiz**: Generate practice questions and quizzes
- **llm-explainer**: Dual-level explanations (beginner + advanced)
- **socratic-llm**: Socratic questioning method
- **exam-prep**: Weighted study plans and exam preparation

## Learning Methodology

See `learn.md` for the 22-technique learning system:
1. Session Setup (Learning Contract, Prompt Engineering)
2. Knowledge Acquisition (Scaffolded, Dual Coding, Examples)
3. Deep Understanding (Socratic, Feynman, Elaborative)
4. Active Practice (Recall, Interleaved, Deliberate)
5. Simulation (Role-Play, Debate)
6. Reflection (Metacognition, SRL Loop, Error Log)
7. Retention (Spaced Repetition, Verification)

## Key Technologies Covered

- **NVIDIA Stack**: NeMo, TensorRT-LLM, Triton, NIM, NCCL, Nsight, TAO, NGC, Base Command
- **Quantization**: FP8, INT8, INT4, AWQ, GPTQ, SmoothQuant
- **Optimization**: Pruning, Knowledge Distillation, Speculative Decoding, 2:4 Sparsity
- **Memory**: Gradient Checkpointing, FlashAttention, PagedAttention
- **Parallelism**: Tensor, Pipeline, Data, Sequence
- **Inference**: vLLM, Continuous Batching, ONNX, TensorRT
- **Fine-tuning**: LoRA, QLoRA, RLHF, DPO, SFT
- **RAG**: Milvus, FAISS, Pinecone, embeddings, reranking

## Logging

All learning activities are tracked in `logs/learning.log` using format:
```
[YYYY-MM-DD HH:MM:SS] [LEVEL] [CATEGORY] Message
```

## Interaction Guidelines

When helping with this knowledge base:
- Reference specific files and line numbers when explaining concepts
- Use the weighted topics to prioritize exam-focused studying
- Apply learning techniques from `learn.md` when teaching
- Provide both beginner and advanced explanations (Dual Coding)
- Include practical examples with NVIDIA tools and commands
- Log learning activities to `logs/learning.log`
