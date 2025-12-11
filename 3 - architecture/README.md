# Group C - Architecture & Applications

## C1. Prompt Engineering (13%)

### Techniques
- Zero-shot
- One-shot
- Few-shot
- Chain-of-Thought (CoT)
- Zero-Shot CoT
- System instructions

### Output Control
- temperature
- top_p
- top_k
- stop sequences
- max_tokens
- JSON / structured outputs

### Advanced Techniques
- Self-consistency
- Prompt chaining
- ReAct (Reasoning + Acting)

**Detailed notes:** [3_1-prompt-engineering.md](3_1-prompt-engineering.md)

---

## C2. LLM Architecture (6%)

### Transformer Components
- Self-attention mechanism
- Multi-head attention (MHA)
- Multi-query attention (MQA)
- Grouped-query attention (GQA)
- Feed-forward network (FFN)
- Layer normalization / RMSNorm

### Positional Encoding
- Absolute positional encoding
- RoPE (Rotary Position Embedding)
- ALiBi (Attention with Linear Biases)

### Activation Functions
- ReLU, GELU, SiLU
- Sigmoid, Tanh

### Architecture Types
- Decoder-only (GPT, Llama)
- Encoder-only (BERT)
- Encoder-decoder (T5, BART)

### Modern Innovations
- FlashAttention
- Mixture of Experts (MoE)
- Sliding window attention

**Detailed notes:** [3_2-llm-architecture.md](3_2-llm-architecture.md)

---

## C3. RAG (Retrieval-Augmented Generation)

### Core Pipeline
1. Ingestion
2. Chunking
3. Embedding (NV-Embed)
4. Indexing (Milvus, FAISS, Pinecone)
5. Retrieval (ANN)
6. Reranking (Cross-Encoder)
7. Generation

### Advanced Patterns
- Hybrid search (dense + sparse)
- Multi-query RAG
- Self-query RAG
- HyDE (Hypothetical Document Embeddings)
- Parent document retrieval

### Function Calling / Tools
- Structured outputs
- External APIs
- Tool use patterns

### NVIDIA Components
- NV-Embed-QA / NV-Embed-v2
- NV-Rerank models
- NeMo Retriever

**Detailed notes:** [3_3-rag.md](3_3-rag.md)

---

## C4. Architectural Nuances

- RoPE (Rotary Position Embedding)
- FlashAttention
- Megatron-Core
- KV Cache optimization
