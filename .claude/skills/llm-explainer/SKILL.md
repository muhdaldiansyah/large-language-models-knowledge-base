---
name: llm-explainer
description: Explain LLM concepts at multiple difficulty levels using Dual Coding technique. Use when someone asks "what is", "explain", "how does X work", or needs clarification on large language model concepts, NVIDIA tools, or ML infrastructure.
allowed-tools: Read, Grep, Glob, Edit
---

# LLM Concept Explainer

Explain any LLM concept using Dual Coding - providing both simple and advanced explanations.

## Instructions

### 1. Dual Coding Format

For every concept, provide TWO explanations:

**Beginner Level**
- Use everyday analogies
- Avoid jargon (or define it immediately)
- Focus on "what" and "why it matters"
- Use visual descriptions

**Advanced Level**
- Technical details and math where relevant
- Implementation specifics
- Trade-offs and edge cases
- Performance considerations

### 2. Explanation Template

```markdown
## [Concept Name]

### Simple Explanation
[Analogy-based explanation a beginner can understand]

### Technical Explanation
[Detailed technical explanation with specifics]

### Key Points
- Point 1
- Point 2
- Point 3

### Related Concepts
- [Related concept 1]
- [Related concept 2]
```

### 3. Example: Explaining Tensor Parallel

**Simple Explanation**
Imagine you have a huge puzzle that's too big for one table. Tensor Parallel is like splitting that puzzle across multiple tables (GPUs), where each table works on different pieces of the same rows. They need to talk to each other frequently to make sure the pieces connect properly.

**Technical Explanation**
Tensor Parallelism splits individual weight matrices across multiple GPUs. For a weight matrix W of shape [H, H], with TP=4, each GPU holds a [H, H/4] slice. During forward pass:
1. Input is broadcast to all GPUs
2. Each GPU computes partial output with its weight slice
3. Results are combined via AllReduce (NCCL)

Requires high-bandwidth interconnect (NVLink preferred: 900 GB/s on H100) due to frequent communication. Best for layers too large for single GPU memory.

### 4. Analogy Library

Use these analogies for common concepts:

| Concept | Analogy |
|---------|---------|
| Quantization | Converting HD video to compressed format - smaller but some quality loss |
| KV-Cache | Taking notes during a conversation so you don't forget what was said |
| Attention | Highlighting important words in a document |
| Embedding | Converting words to GPS coordinates in meaning-space |
| Fine-tuning | Teaching a general doctor to become a specialist |
| LoRA | Adding sticky notes to a textbook instead of rewriting it |
| RAG | Open-book exam vs closed-book |
| Batch Processing | Waiting for a full bus before departing vs taxi per person |
| PagedAttention | Virtual memory for GPUs |
| NVLink | High-speed highway between GPUs vs regular roads (PCIe) |

### 5. Visual Representations

When helpful, include ASCII diagrams:

```
Pipeline Parallel (PP):
GPU 0: [Layer 1-10] → GPU 1: [Layer 11-20] → GPU 2: [Layer 21-30]
         ↓                    ↓                     ↓
      Output 1            Output 2              Final Output

Tensor Parallel (TP):
         Input
           ↓
    ┌──────┼──────┐
   GPU0  GPU1  GPU2  (each has weight slice)
    └──────┼──────┘
        AllReduce
           ↓
        Output
```

### 6. Context Bridges

Connect new concepts to familiar ones:
- "This is similar to X, but..."
- "Unlike X which does Y, this..."
- "Think of it as X meets Y"

### 7. When to Use Each Level

- Start with **beginner** for first-time explanations
- Use **advanced** when user asks "how exactly" or "technically"
- Offer both when user's level is unclear

### 8. Knowledge Base Reference

| Topic | File |
|-------|------|
| Model Optimization | `1 - infrastructure/1_1-model-optimization.md` |
| GPU Acceleration | `1 - infrastructure/1_2-gpu-acceleration.md` |
| Deployment | `1 - infrastructure/1_3-deployment.md` |
| NVIDIA NIM | `1 - infrastructure/1_4-nvidia-nim.md` |
| Fine-Tuning | `2 - data-training/2_1-fine-tuning.md` |
| Data Preparation | `2 - data-training/2_2-data-preparation.md` |
| Data Curation | `2 - data-training/2_3-data-curation.md` |
| Prompt Engineering | `3 - architecture/3_1-prompt-engineering.md` |
| LLM Architecture | `3 - architecture/3_2-llm-architecture.md` |
| RAG | `3 - architecture/3_3-rag.md` |
| Evaluation | `4 - operations/4_1-evaluation.md` |
| Monitoring | `4 - operations/4_2-monitoring.md` |
| Safety | `4 - operations/4_3-safety.md` |
| ML Fundamentals | `5 - fundamentals/5_1-ml-fundamentals.md` |
| NeMo 2.0 | `5 - fundamentals/5_2-nemo-2.md` |
