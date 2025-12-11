# 3.2 LLM Architecture (6%)

## Overview

Large Language Models are built on the **Transformer** architecture, introduced in "Attention Is All You Need" (2017).

```
Input Tokens → Embedding → [Transformer Blocks × N] → Output Logits → Tokens
```

---

## Transformer Components

### Self-Attention Mechanism

The core innovation: allows each token to "attend" to all other tokens in the sequence.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Components:**
- **Q (Query):** What am I looking for?
- **K (Key):** What do I contain?
- **V (Value):** What information do I provide?
- **d_k:** Dimension of keys (scaling factor)

**Step-by-step:**
1. Create Q, K, V matrices from input embeddings
2. Compute attention scores: `QK^T`
3. Scale by `√d_k` to prevent vanishing gradients
4. Apply softmax to get attention weights
5. Multiply weights by V to get output

### Multi-Head Attention (MHA)

Run multiple attention operations in parallel, each learning different relationships.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Benefits:**
- Different heads capture different patterns
- Head 1 might learn syntax
- Head 2 might learn semantics
- Head 3 might learn positional relationships

**Typical configurations:**
| Model Size | Heads | d_model | d_k (per head) |
|------------|-------|---------|----------------|
| Small | 8 | 512 | 64 |
| Base | 12 | 768 | 64 |
| Large | 16 | 1024 | 64 |
| XL | 32 | 2048 | 64 |

### Multi-Query Attention (MQA)

Optimization: share K and V across all heads, only Q is per-head.

```
Standard MHA: Each head has own Q, K, V
MQA: Each head has own Q, shared K, V
```

**Benefits:**
- Reduces KV cache size by `num_heads` factor
- Faster inference
- Slight accuracy tradeoff

### Grouped-Query Attention (GQA)

Middle ground: group heads to share K, V.

```
MHA: 32 heads, 32 KV pairs
GQA: 32 heads, 8 KV pairs (4 heads share each KV)
MQA: 32 heads, 1 KV pair
```

**Used in:** Llama 2 70B, Llama 3, Mistral

---

## Feed-Forward Network (FFN)

Applied after attention in each transformer block.

```python
FFN(x) = activation(x × W_1 + b_1) × W_2 + b_2
```

**Typical structure:**
- Expand: d_model → 4 × d_model
- Activation function
- Contract: 4 × d_model → d_model

**SwiGLU variant (modern LLMs):**
```python
SwiGLU(x) = (x × W_1 × σ(x × W_gate)) × W_2
```

---

## Layer Normalization

Stabilizes training by normalizing activations.

### Pre-LayerNorm (Modern)
```
x → LayerNorm → Attention → + → LayerNorm → FFN → +
↑_____________________________↑     ↑_______________↑
```

### Post-LayerNorm (Original)
```
x → Attention → + → LayerNorm → FFN → + → LayerNorm
    ↑__________↑                ↑_____↑
```

### RMSNorm (Efficient)

Simplified normalization without mean centering:
```
RMSNorm(x) = x / √(mean(x²) + ε) × γ
```

**Used in:** Llama, Mistral, most modern LLMs

---

## Positional Encoding

Transformers have no inherent notion of position. Positional encodings add this information.

### Absolute Positional Encoding (Original)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Limitation:** Fixed maximum sequence length

### Rotary Position Embedding (RoPE)

Encodes position through rotation of query and key vectors.

```
RoPE(x, pos) = x × R(pos)

where R(pos) is a rotation matrix based on position
```

**Benefits:**
- Relative position awareness
- Better length generalization
- Decays attention with distance naturally

**Used in:** Llama, Mistral, most modern LLMs

### ALiBi (Attention with Linear Biases)

Add linear bias to attention scores based on distance:
```
Attention_ALiBi = softmax(QK^T - m × |i - j|) × V
```

**Benefits:**
- No learned positional parameters
- Excellent length extrapolation
- Used in: BLOOM, MPT

---

## Activation Functions

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- Simple, fast
- "Dying ReLU" problem

### GELU (Gaussian Error Linear Unit)
```
GELU(x) = x × Φ(x)  # Φ is CDF of standard normal
≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```
- Smooth approximation of ReLU
- Used in: BERT, GPT-2, GPT-3

### SiLU / Swish
```
SiLU(x) = x × σ(x)  # σ is sigmoid
```
- Self-gated activation
- Used in: Llama, modern LLMs

### Comparison
| Function | Formula | Used In |
|----------|---------|---------|
| ReLU | max(0, x) | Older models |
| GELU | x × Φ(x) | BERT, GPT |
| SiLU | x × σ(x) | Llama, Mistral |

---

## Architecture Types

### Decoder-Only (Autoregressive)

```
[Input tokens] → Decoder → [Next token prediction]
                   ↓
            Causal masking (can only see past)
```

**Characteristics:**
- Unidirectional attention (left-to-right)
- Each token only attends to previous tokens
- Used for text generation

**Examples:** GPT, Llama, Mistral, Claude

### Encoder-Only (Bidirectional)

```
[Input tokens] → Encoder → [Contextualized embeddings]
                   ↓
            Full attention (sees all tokens)
```

**Characteristics:**
- Bidirectional attention
- Each token sees entire sequence
- Used for understanding/classification

**Examples:** BERT, RoBERTa, DeBERTa

### Encoder-Decoder (Seq2Seq)

```
[Source] → Encoder → [Hidden state] → Decoder → [Target]
              ↓                          ↓
        Bidirectional              Cross-attention to encoder
```

**Characteristics:**
- Encoder: bidirectional attention on input
- Decoder: causal attention + cross-attention to encoder
- Used for translation, summarization

**Examples:** T5, BART, mT5

### Architecture Comparison

| Type | Attention | Use Case | Examples |
|------|-----------|----------|----------|
| Decoder-only | Causal | Generation | GPT, Llama |
| Encoder-only | Bidirectional | Understanding | BERT |
| Encoder-Decoder | Both | Seq2Seq | T5, BART |

---

## Modern Architectural Innovations

### FlashAttention

Memory-efficient attention implementation.

```
Standard: O(N²) memory for attention matrix
FlashAttention: O(N) memory using tiling
```

**Key techniques:**
- Tiling: compute attention in blocks
- Recomputation: trade compute for memory
- Kernel fusion: reduce memory bandwidth

**Benefits:**
- 2-4x faster training
- Enables longer sequences
- No approximation (exact attention)

### Mixture of Experts (MoE)

Replace FFN with multiple "expert" FFNs, route tokens to subset.

```
Token → Router → Select top-k experts → Combine outputs
                      ↓
              Expert 1, Expert 2, ... Expert N
```

**Benefits:**
- More parameters without proportional compute
- Sparse activation (only k experts per token)

**Examples:** Mixtral 8x7B (8 experts, 2 active)

### Sliding Window Attention

Limit attention to local window for efficiency.

```
Full attention: O(N²) - every token to every token
Window attention: O(N × W) - each token to W neighbors
```

**Used in:** Mistral, Longformer

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Self-Attention | softmax(QK^T / √d_k) × V |
| Parameters (approx) | 12 × L × d² (L=layers, d=d_model) |
| FLOPs per token | ~2 × Parameters |
| KV Cache size | 2 × L × d × seq_len × batch × precision |

---

## NVIDIA-Specific Optimizations

### Megatron-Core

NVIDIA's library for training large transformer models:
- Tensor parallelism for attention and FFN
- Pipeline parallelism across layers
- Sequence parallelism for long contexts
- Optimized kernels for A100/H100

### TensorRT-LLM Optimizations

- FlashAttention integration
- FP8 computation on Hopper
- Fused kernels for transformer blocks
- In-flight batching support
