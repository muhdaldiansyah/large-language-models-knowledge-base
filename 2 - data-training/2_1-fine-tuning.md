# 2.1 Fine-Tuning (13%)

## Overview

Fine-tuning adapts a pre-trained model to specific tasks or domains by updating model weights on task-specific data.

```
Pre-trained Model → Fine-tuning on Task Data → Specialized Model
     (General)           (Your Data)              (Task-Specific)
```

---

## Fine-Tuning Methods

### Full Fine-Tuning

- Updates ALL model parameters
- Highest accuracy potential
- Requires significant compute and memory
- Risk of catastrophic forgetting

```
Parameters updated: 100% (billions for large models)
Memory: Full model + optimizer states + gradients
Use case: When you have lots of data and compute
```

### LoRA (Low-Rank Adaptation)

- Freezes original weights
- Adds small trainable rank decomposition matrices
- Dramatically reduces trainable parameters

```
Original weight: W (d × k)
LoRA: W + BA where B (d × r), A (r × k), r << min(d,k)

Example: r=8 for a 4096×4096 matrix
- Original: 16.7M parameters
- LoRA: 65K parameters (0.4%)
```

**Key Parameters:**
- `r` (rank): Typically 8-64, higher = more capacity
- `alpha`: Scaling factor, usually = r or 2×r
- `target_modules`: Which layers to adapt (q_proj, v_proj, etc.)

```python
# Example LoRA config
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05
}
```

### QLoRA (Quantized LoRA)

- Base model quantized to 4-bit (NF4)
- LoRA adapters in higher precision
- Enables fine-tuning 65B models on single GPU

```
Memory comparison (65B model):
- Full fine-tuning: ~780 GB
- LoRA (FP16): ~130 GB
- QLoRA (4-bit): ~48 GB
```

**Key Components:**
- 4-bit NormalFloat (NF4) quantization
- Double quantization (quantize the quantization constants)
- Paged optimizers (offload to CPU when needed)

### P-Tuning / Prompt Tuning

- Freezes entire model
- Only trains soft prompt embeddings
- Even fewer parameters than LoRA

```
Input: [SOFT_TOKENS] + [USER_INPUT]
        (learned)      (fixed)

Trainable params: num_virtual_tokens × hidden_dim
Example: 20 tokens × 4096 = 81,920 parameters
```

### Adapters

- Small bottleneck modules inserted between layers
- Original weights frozen
- Can stack multiple adapters for different tasks

```
Layer output → Adapter(down → nonlinear → up) → + residual
                 (d → r)              (r → d)
```

---

## Training Paradigms

### SFT (Supervised Fine-Tuning)

Standard supervised learning on input-output pairs.

```
Dataset format:
{"input": "Translate to French: Hello", "output": "Bonjour"}
{"input": "Summarize: [long text]", "output": "[summary]"}
```

**Loss:** Cross-entropy on output tokens only

### RLHF (Reinforcement Learning from Human Feedback)

Three-stage process:

```
Stage 1: SFT
Pre-trained → SFT on demonstrations → SFT Model

Stage 2: Reward Model Training
Collect comparisons → Train reward model
Human ranks: Response A > Response B > Response C

Stage 3: RL Fine-tuning (PPO)
SFT Model + Reward Model → PPO training → RLHF Model
```

**Components:**
- Policy model (being trained)
- Reference model (frozen SFT model for KL penalty)
- Reward model (predicts human preference)
- Value model (estimates expected reward)

**PPO Objective:**
```
maximize E[reward(response)] - β × KL(policy || reference)
```

### DPO (Direct Preference Optimization)

- Simpler alternative to RLHF
- No separate reward model needed
- Directly optimizes on preference pairs

```
Dataset format:
{
  "prompt": "Write a poem about AI",
  "chosen": "Silicon dreams dance...",    # preferred
  "rejected": "AI is computers..."        # not preferred
}
```

**DPO Loss:**
```
L = -log σ(β × (log π(chosen) - log π(rejected)
              - log π_ref(chosen) + log π_ref(rejected)))
```

**Advantages over RLHF:**
- No reward model training
- No RL instabilities
- Simpler implementation
- Often comparable results

---

## Training Challenges

### Vanishing Gradients

- Gradients become tiny in deep networks
- Earlier layers stop learning
- Common in RNNs, deep networks

**Solutions:**
- Residual connections (skip connections)
- Layer normalization
- Careful initialization
- Gradient clipping

### Exploding Gradients

- Gradients become extremely large
- Causes NaN losses, unstable training

**Solutions:**
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Lower learning rate
- Gradient scaling (for mixed precision)

---

## NeMo Fine-Tuning

### Configuration Example

```yaml
# NeMo fine-tuning config
model:
  restore_from_path: /path/to/base_model.nemo

  peft:
    peft_scheme: "lora"
    lora_tuning:
      adapter_dim: 32
      alpha: 32
      target_modules: ["attention_qkv"]

trainer:
  devices: 8
  num_nodes: 1
  precision: bf16
  max_steps: 1000

data:
  train_ds:
    file_path: /data/train.jsonl
    micro_batch_size: 4
    global_batch_size: 32
```

### NeMo 2.0 Changes

```python
# NeMo 2.0 Python-based config
from nemo.collections.llm import finetune, LoRA

finetune(
    model=AutoModel("llama-3-8b"),
    data=SquadDataModule(),
    peft=LoRA(dim=32, alpha=32),
    trainer=Trainer(devices=8, precision="bf16"),
)
```

---

## Comparison Table

| Method | Trainable Params | Memory | Accuracy | Speed |
|--------|-----------------|--------|----------|-------|
| Full FT | 100% | Very High | Highest | Slow |
| LoRA | 0.1-1% | Medium | High | Fast |
| QLoRA | 0.1-1% | Low | High | Medium |
| P-Tuning | <0.01% | Very Low | Medium | Very Fast |
| Adapters | 1-5% | Medium | High | Fast |

---

## Best Practices

1. **Start with LoRA** for most use cases
2. **Use QLoRA** when GPU memory is limited
3. **Full fine-tuning** only with abundant data and compute
4. **DPO over RLHF** for preference alignment (simpler)
5. **Monitor validation loss** to detect overfitting
6. **Use warmup** for learning rate scheduling
7. **Save checkpoints** frequently
