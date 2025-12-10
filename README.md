# Large Language Models Knowledge Base

A comprehensive reference for production LLM systems using the NVIDIA ecosystem.

## Topics

### 1 - Infrastructure
- **1.1. Model Optimization** - Quantization (FP8, INT8, INT4), TensorRT-LLM, mixed precision
- **1.2. GPU Acceleration** - Tensor/Pipeline/Data parallelism, NVLink, NCCL, Tensor Cores
- **1.3. Deployment** - Triton Inference Server, Kubernetes, Docker, dynamic batching
- **1.4. NVIDIA NIM** - Day-1 deployment, OpenAI-compatible API

### 2 - Data & Training
- **2.1. Fine-Tuning** - LoRA, QLoRA, P-Tuning, SFT, RLHF, DPO
- **2.2. Data Preparation** - Tokenization (BPE, WordPiece), RAPIDS ecosystem
- **2.3. Data Curation** - Deduplication, quality filtering, PII redaction

### 3 - Architecture & Applications
- **3.1. Prompt Engineering** - Zero/few-shot, Chain-of-Thought, temperature, structured outputs
- **3.2. LLM Architecture** - Transformers, attention mechanisms, positional encoding
- **3.3. RAG** - Embeddings, vector databases, chunking, reranking
- **3.4. Architectural Nuances** - RoPE, FlashAttention, Megatron-Core

### 4 - Operations
- **4.1. Evaluation** - Perplexity, BLEU, ROUGE, benchmarking
- **4.2. Monitoring** - Latency metrics, throughput, GPU utilization
- **4.3. Safety** - NeMo Guardrails, jailbreak detection, PII masking
- **4.4. NVIDIA Tools** - NeMo, TensorRT-LLM, Triton, Nsight, ModelOpt

### 5 - Fundamentals
- **5.1. Additional Topics** - ML basics, NeMo 2.0, Slurm, Kubernetes, DGX

## Key Technologies

NeMo | TensorRT-LLM | Triton | RAPIDS | Milvus | FAISS

## Resources

- [NVIDIA NeMo](https://developer.nvidia.com/nemo)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Inference Server](https://github.com/triton-inference-server)
