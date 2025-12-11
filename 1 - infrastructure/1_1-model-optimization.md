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

Pruning removes unnecessary weights/neurons to reduce model size and computation.

### Unstructured Pruning

Removes individual weights (sparse matrices).

```
Original weights:    Pruned (50% sparse):
[0.5, 0.1, 0.8]     [0.5, 0.0, 0.8]
[0.2, 0.9, 0.3]  →  [0.0, 0.9, 0.0]
[0.7, 0.4, 0.6]     [0.7, 0.0, 0.6]
```

**Methods:**
- **Magnitude Pruning:** Remove smallest weights
- **Movement Pruning:** Prune based on weight changes during training
- **Lottery Ticket Hypothesis:** Find sparse subnetworks

```python
def magnitude_prune(weights, sparsity=0.5):
    """Prune smallest magnitude weights"""
    threshold = np.percentile(np.abs(weights), sparsity * 100)
    mask = np.abs(weights) > threshold
    return weights * mask
```

**Challenges:**
- Requires sparse matrix support for speedup
- Hardware often optimized for dense operations
- NVIDIA Ampere+ supports 2:4 structured sparsity

### Structured Pruning

Removes entire structures (neurons, heads, layers).

```
Original model:              After structured pruning:
[Head1, Head2, Head3, Head4] → [Head1, Head3, Head4]
[Layer1, Layer2, Layer3]     → [Layer1, Layer3]
```

**Types:**
| Type | What's Removed | Speedup |
|------|----------------|---------|
| Channel pruning | Entire filter channels | Direct |
| Head pruning | Attention heads | Direct |
| Layer pruning | Entire transformer layers | Direct |
| Block pruning | N×M weight blocks | Hardware-dependent |

**Head Pruning Example:**
```python
def compute_head_importance(model, eval_data):
    """Score attention heads by importance"""
    head_importance = {}
    for layer_idx, layer in enumerate(model.layers):
        for head_idx in range(num_heads):
            importance = compute_taylor_score(layer, head_idx, eval_data)
            head_importance[(layer_idx, head_idx)] = importance
    return head_importance

def prune_heads(model, head_importance, prune_ratio=0.3):
    """Remove least important heads"""
    sorted_heads = sorted(head_importance.items(), key=lambda x: x[1])
    heads_to_prune = sorted_heads[:int(len(sorted_heads) * prune_ratio)]
    for (layer_idx, head_idx), _ in heads_to_prune:
        model.layers[layer_idx].remove_head(head_idx)
    return model
```

### 2:4 Structured Sparsity (NVIDIA)

Hardware-accelerated sparsity on Ampere+ GPUs.

```
2:4 Pattern: In every 4 consecutive elements, 2 must be zero

Valid:   [0, 0, 0.5, 0.3]  ✓
Valid:   [0.1, 0, 0, 0.8]  ✓
Invalid: [0.1, 0.2, 0, 0.8] ✗ (only 1 zero in group)
```

**Benefits:**
- 2x speedup with Tensor Cores
- ~50% memory reduction
- Minimal accuracy loss with fine-tuning

```python
# TensorRT-LLM 2:4 sparsity
from tensorrt_llm.quantization import sparse_weights

model = sparse_weights(
    model,
    sparsity_pattern="2:4",
    calibration_data=calib_dataset
)
```

### Pruning + Fine-tuning Pipeline

```
Pre-trained Model → Prune → Fine-tune → Evaluate → Iterate
                      ↓
              Remove X% of weights
                      ↓
              Retrain to recover accuracy
```

---

## Knowledge Distillation

Transfer knowledge from a large "teacher" model to a smaller "student" model.

```
Teacher (Large)          Student (Small)
    70B parameters  →      7B parameters
    High accuracy          ~Similar accuracy
    Slow inference         Fast inference
```

### Response-Based Distillation

Student learns to match teacher's output probabilities.

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=2.0, alpha=0.5):
    """Combine soft (distillation) and hard (ground truth) losses"""
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence for soft targets
    distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    distill_loss = distill_loss * (temperature ** 2)

    # Cross-entropy for hard targets
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * distill_loss + (1 - alpha) * hard_loss
```

**Temperature Effect:**
```
Temperature = 1.0: Sharp distribution (confident predictions)
Temperature = 2.0: Softer distribution (more info transfer)
Temperature = 5.0: Very soft (may lose discrimination)
```

### Feature-Based Distillation

Student learns to match teacher's intermediate representations.

```python
def feature_distillation_loss(student_features, teacher_features):
    """Match intermediate layer representations"""
    if student_features.shape != teacher_features.shape:
        student_features = projection_layer(student_features)
    return F.mse_loss(student_features, teacher_features)
```

### LLM-Specific Distillation

#### On-Policy Distillation

Generate data with teacher, train student on it.

```python
def on_policy_distillation(teacher, student, prompts):
    training_data = []
    for prompt in prompts:
        teacher_response = teacher.generate(prompt, temperature=0.7)
        training_data.append({"prompt": prompt, "response": teacher_response})
    student.finetune(training_data)
```

### Distillation Best Practices

| Aspect | Recommendation |
|--------|----------------|
| Temperature | 2-4 for LLMs |
| Alpha (soft/hard) | 0.5-0.9 (favor soft targets) |
| Student size | 10-30% of teacher |
| Data | Diverse, high-quality prompts |
| Initialization | From smaller pre-trained model |

### NeMo Distillation

```python
from nemo.collections.llm import distill

student = distill(
    teacher_model="nvidia/nemotron-4-340b",
    student_config=GPTConfig(hidden_size=2048, num_layers=24),
    data_path="/data/distillation_prompts.jsonl",
    temperature=3.0,
    alpha=0.7
)
```

---

## Speculative Decoding

Accelerate inference using a small "draft" model to predict tokens, then verify with the main model.

```
Traditional:  Token 1 → Token 2 → Token 3 → ... (sequential, slow)

Speculative:  Draft predicts [t1,t2,t3,t4] → Main verifies all (parallel)
```

### How It Works

```
Step 1: Draft model generates K candidate tokens quickly
        Draft: "The capital of France is" → ["Paris", ".", "It", "is"]

Step 2: Main model verifies all K tokens in single forward pass
        Main model scores: [✓ Paris, ✓ ., ✗ It, -]

Step 3: Accept verified tokens, reject from first mismatch
        Accepted: "Paris ."

Step 4: Main model generates correct token for rejected position
        Generate: "The" (instead of "It")

Step 5: Repeat from new position
```

### Implementation

```python
def speculative_decode(main_model, draft_model, prompt, k=4):
    """Speculative decoding with draft model"""
    generated = prompt

    while not is_done(generated):
        # Draft model generates k tokens quickly
        draft_tokens = []
        draft_probs = []
        for _ in range(k):
            logits = draft_model(generated + draft_tokens)
            token = sample(logits)
            draft_tokens.append(token)
            draft_probs.append(softmax(logits)[token])

        # Main model verifies all positions in ONE forward pass
        main_logits = main_model(generated, draft_tokens)
        main_probs = softmax(main_logits)

        # Verify each token
        accepted = []
        for i, token in enumerate(draft_tokens):
            p_main = main_probs[i, token]
            p_draft = draft_probs[i]

            if random.random() < min(1, p_main / p_draft):
                accepted.append(token)
            else:
                # Sample from adjusted distribution
                adjusted = relu(main_probs[i] - draft_probs)
                accepted.append(sample(adjusted / adjusted.sum()))
                break

        generated += accepted

    return generated
```

### Speedup Analysis

```
Typical speedup: 2-3x for well-matched draft models

Factors:
- Draft model speed (faster = better)
- Acceptance rate (higher = better)
- Speculation length k (4-8 typically optimal)
```

### Draft Model Selection

| Main Model | Recommended Draft | Notes |
|------------|-------------------|-------|
| Llama 3.1 70B | Llama 3.1 8B | Same family, high acceptance |
| Mixtral 8x22B | Mistral 7B | Compatible architecture |

### TensorRT-LLM Speculative Decoding

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    speculative_config={
        "draft_model": "meta-llama/Llama-3.1-8B",
        "num_speculative_tokens": 4,
        "use_draft_logits": True
    }
)

output = llm.generate(
    "The capital of France is",
    sampling_params=SamplingParams(max_tokens=100)
)
```

### Medusa: Multi-Head Speculation

Add multiple prediction heads to generate several tokens per position.

```
Hidden state → Head 1 → Token +1 prediction
            → Head 2 → Token +2 prediction
            → Head 3 → Token +3 prediction
```

### Speculative Decoding Best Practices

1. **Match architectures:** Draft from same model family
2. **Tune k:** 4-8 tokens typically optimal
3. **Monitor acceptance rate:** Should be >70%
4. **Consider memory:** Draft adds memory overhead
5. **Batch size:** Less effective with large batches
