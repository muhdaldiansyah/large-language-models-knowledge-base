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
