# 1.1 Model Optimization (17%)

## Quantization Methods

### FP8
- 8-bit floating point format
- Supported on Hopper architecture (H100)
- Good balance of speed and accuracy

### INT8 SmoothQuant
- Migrates quantization difficulty from activations to weights
- Better accuracy than naive INT8
- Requires calibration dataset

### INT4 AWQ / GPTQ
- **AWQ** - Activation-aware Weight Quantization
- **GPTQ** - Post-training quantization using approximate second-order information
- Aggressive compression with acceptable accuracy loss

### W4A8
- 4-bit weights, 8-bit activations
- Hybrid approach for memory efficiency

---

## TensorRT-LLM Build Flags

- `--max_batch_size`
- `--max_input_len`
- `--max_output_len`
- `--use_inflight_batching`
- `--paged_kv_cache`
- `--quantization` (fp8, int8, int4)

---

## Key Concepts

### AMAX Calibration
- Determines scaling factors for quantization
- Uses representative dataset
- Critical for FP8 accuracy

### KV-Cache Quantization
- Reduces memory footprint of key-value cache
- Enables longer context lengths
- FP8 or INT8 KV cache options

### In-flight Batching
- Dynamic request batching during inference
- Maximizes GPU utilization
- Reduces latency for variable-length requests

### PagedAttention
- Memory management technique from vLLM
- Eliminates memory fragmentation
- Enables efficient KV cache allocation

### PTQ vs QAT
- **PTQ** - Post-Training Quantization (faster, no retraining)
- **QAT** - Quantization-Aware Training (better accuracy, requires training)

---

## Mixed Precision

### FP16
- 16-bit floating point
- Standard for inference
- Good speed/accuracy tradeoff

### BF16
- Brain floating point 16
- Better dynamic range than FP16
- Preferred for training

### AMP Optimization Levels
- **O0** - FP32 training (baseline)
- **O1** - Mixed precision (recommended)
- **O2** - Almost FP16 (faster, slight accuracy loss)

---

## Compute Estimation

### FLOPs Calculation
- Forward pass: ~2 * Parameters * Tokens
- Training: ~6 * Parameters * Tokens (forward + backward + optimizer)

### Activation Memory
- Proportional to batch size and sequence length
- Checkpoint activations to reduce memory

### Tokens/sec Formulas
- Throughput = Batch Size * Sequence Length / Latency
- Time per token = Latency / Output Tokens

---

## Pruning

Removing unnecessary weights to reduce model size and improve inference speed.

### Pruning Types

#### Unstructured Pruning
- Removes individual weights (sets to zero)
- High sparsity achievable (90%+)
- Requires sparse matrix support for speedup

```python
# Magnitude-based pruning
import torch.nn.utils.prune as prune

# Prune 30% of weights with lowest magnitude
prune.l1_unstructured(layer, name='weight', amount=0.3)

# Make pruning permanent
prune.remove(layer, 'weight')
```

#### Structured Pruning
- Removes entire structures (neurons, heads, layers)
- Direct speedup without sparse matrix support
- Lower sparsity but better hardware efficiency

```python
# Prune 20% of channels
prune.ln_structured(layer, name='weight', amount=0.2, n=2, dim=0)

# Attention head pruning
def prune_attention_heads(model, heads_to_prune):
    """Remove specified attention heads"""
    for layer_idx, head_indices in heads_to_prune.items():
        layer = model.layers[layer_idx].attention
        mask = torch.ones(layer.num_heads)
        mask[head_indices] = 0
        layer.head_mask = mask
```

### Pruning Strategies

| Strategy | Description | Speedup | Accuracy Impact |
|----------|-------------|---------|-----------------|
| Magnitude | Remove smallest weights | Medium | Low |
| Movement | Remove by gradient × weight | Medium | Lower |
| Lottery Ticket | Find sparse subnetwork | High | Low |
| Layer-wise | Different ratios per layer | High | Medium |

### SparseGPT

State-of-the-art one-shot pruning for LLMs:

```python
# SparseGPT approach (conceptual)
# 1. Compute Hessian approximation for each layer
# 2. Solve for optimal sparse weights
# 3. Update remaining weights to compensate

# Can achieve 50-60% sparsity with minimal accuracy loss
# Works in single forward pass through calibration data
```

### NVIDIA 2:4 Structured Sparsity

Hardware-accelerated sparsity on Ampere/Hopper GPUs:

```python
# 2:4 pattern: 2 zeros in every 4 consecutive weights
from apex.contrib.sparsity import ASP

# Prepare model for 2:4 sparsity
ASP.prune_trained_model(model, optimizer)

# Compute sparse masks
ASP.compute_sparse_masks()

# Fine-tune with sparsity
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**Benefits:**
- 2x Tensor Core speedup on A100/H100
- Native hardware support (no custom kernels)
- Minimal accuracy loss (~1%)

---

## Knowledge Distillation

Transfer knowledge from large (teacher) model to smaller (student) model.

### Distillation Types

#### Response-based Distillation
Match teacher's output probabilities:

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=2.0, alpha=0.5):
    """
    Combine soft targets (teacher) and hard targets (labels)
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
    soft_loss = soft_loss * (temperature ** 2)

    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

#### Feature-based Distillation
Match intermediate representations:

```python
def feature_distillation_loss(student_hidden, teacher_hidden):
    """Match hidden states between student and teacher"""
    # Project student to teacher dimension if needed
    if student_hidden.shape != teacher_hidden.shape:
        student_hidden = projection_layer(student_hidden)

    return F.mse_loss(student_hidden, teacher_hidden)
```

### Distillation Best Practices

| Aspect | Recommendation |
|--------|----------------|
| Temperature | 2-4 for soft targets |
| Alpha | 0.5-0.7 (balance soft/hard) |
| Student size | 4-10x smaller than teacher |
| Training data | Use teacher to generate labels |
| Layer matching | Map student layers to teacher |

---

## Speculative Decoding

Use small draft model to predict multiple tokens, verify with large model in parallel.

### How It Works

```
Traditional decoding (sequential):
Large Model → Token 1 → Large Model → Token 2 → Large Model → Token 3
              (slow)                   (slow)                   (slow)

Speculative decoding (parallel verification):
Draft Model → [Token 1, 2, 3, 4, 5] → Large Model verifies all at once
   (fast)                                    (single forward pass)
                                               ↓
                               Accept: [1, 2, 3] Reject: [4, 5]
                                               ↓
                               Output: Token 1, 2, 3 + continue
```

### Implementation Concept

```python
def speculative_decode(prompt, draft_model, target_model, k=5):
    """
    k: number of tokens to draft ahead
    """
    generated = prompt

    while not done:
        # Draft k tokens with small model (fast)
        draft_tokens = draft_model.generate(generated, num_tokens=k)

        # Verify all k tokens with target model (single forward pass)
        target_logits = target_model.forward(
            generated + draft_tokens,
            return_all_logits=True
        )

        # Accept tokens where target agrees with draft
        accepted = verify_tokens(draft_tokens, target_logits)
        generated.extend(accepted)

    return generated
```

### Performance Characteristics

| Aspect | Value |
|--------|-------|
| Speedup | 2-3x typical |
| Draft model size | 10-20% of target |
| Optimal k (lookahead) | 4-8 tokens |
| Best for | Long generations, high-latency models |
| Accuracy | Mathematically identical to target |

### TensorRT-LLM Speculative Decoding

```python
# Enable in TensorRT-LLM build
trtllm-build \
    --model llama-70b \
    --speculative_decoding_mode draft_model \
    --draft_model llama-8b \
    --num_draft_tokens 5
```

### Medusa: Multi-Head Speculation

Train additional prediction heads instead of separate draft model:

```
Original LLM output → Head 0: token t+1
                    → Head 1: token t+2
                    → Head 2: token t+3

All heads verified with tree attention in single pass
```

---

## Model Compression Summary

| Technique | Memory Reduction | Speedup | Accuracy Loss | Best For |
|-----------|------------------|---------|---------------|----------|
| FP8 Quantization | 2x | 2x | <1% | H100 inference |
| INT8 SmoothQuant | 2x | 1.5-2x | 1-2% | A100/H100 |
| INT4 (AWQ/GPTQ) | 4x | 2-3x | 1-3% | Memory-limited |
| 2:4 Sparsity | 2x | 2x | ~1% | A100/H100 |
| Distillation | 4-10x | 4-10x | 2-5% | Smaller deployment |
| Speculative | 1x | 2-3x | 0% | Long generation |
| Combined | 8-20x | 5-10x | 3-8% | Edge deployment |

---

## Troubleshooting Optimization

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Accuracy drop after quantization | Perplexity +5% | Use calibration data, try AWQ over GPTQ |
| No speedup from sparsity | Same latency | Verify hardware support, use 2:4 structured |
| OOM during quantization | CUDA OOM | Quantize layer-by-layer, reduce batch size |
| Distillation not converging | High loss | Increase temperature, verify layer mapping |
| Speculative low acceptance | <50% accept rate | Use better draft model, tune k value |
| INT4 quality issues | Incoherent outputs | Use group quantization, increase group size |
