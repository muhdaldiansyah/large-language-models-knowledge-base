# 1.5 NVIDIA AI Ecosystem

## Overview

NVIDIA provides a comprehensive ecosystem of tools, platforms, and services for developing and deploying LLMs at scale.

```
NVIDIA AI Ecosystem:
┌─────────────────────────────────────────────────────────────┐
│                   NVIDIA AI Enterprise                       │
├─────────────────────────────────────────────────────────────┤
│  NGC Catalog  │  NIM  │  Base Command  │  DGX Cloud         │
├─────────────────────────────────────────────────────────────┤
│  NeMo  │  TensorRT-LLM  │  Triton  │  TAO  │  RAPIDS        │
├─────────────────────────────────────────────────────────────┤
│                    CUDA / cuDNN / NCCL                       │
└─────────────────────────────────────────────────────────────┘
```

---

## NGC (NVIDIA GPU Cloud) Catalog

Central repository for GPU-optimized software, models, and containers.

### NGC Components

| Component | Description |
|-----------|-------------|
| Containers | Pre-built Docker images for AI/ML |
| Models | Pre-trained models (LLMs, vision, speech) |
| Helm Charts | Kubernetes deployment templates |
| Resources | Jupyter notebooks, scripts |

### NGC CLI Usage

```bash
# Install NGC CLI
pip install ngc-cli

# Configure
ngc config set

# List available containers
ngc registry image list nvidia/nemo

# Pull container
docker pull nvcr.io/nvidia/nemo:24.05

# List models
ngc registry model list nvidia/nemo/*

# Download model
ngc registry model download-version nvidia/nemo/megatron_gpt_345m:1.0
```

### NGC Container Registry

```bash
# Login to NGC
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY

# Common containers
nvcr.io/nvidia/nemo:24.05              # NeMo framework
nvcr.io/nvidia/tritonserver:24.05-py3  # Triton Inference Server
nvcr.io/nvidia/tensorrt:24.05-py3      # TensorRT
nvcr.io/nvidia/pytorch:24.05-py3       # PyTorch optimized
nvcr.io/nim/meta/llama-3.1-8b-instruct # NIM container

# Pull and run
docker run --gpus all -it nvcr.io/nvidia/nemo:24.05 bash
```

### NGC Models for LLMs

| Model | Parameters | Use Case |
|-------|------------|----------|
| Nemotron-4-340B | 340B | General-purpose |
| Llama-3.1 (NeMo) | 8B-405B | Instruction following |
| Mistral NeMo | 12B | Efficient inference |
| NV-Embed | - | Text embeddings |
| NV-Rerank | - | Reranking for RAG |

---

## NVIDIA TAO Toolkit

Transfer learning toolkit for customizing pre-trained models.

### TAO Overview

```
Pre-trained Model → TAO Toolkit → Domain-Specific Model
                         ↓
                    Fine-tuning
                    Pruning
                    Quantization
                    Export
```

### TAO for LLMs

```bash
# Install TAO CLI
pip install nvidia-tao

# Configure
tao model config --model_name llama-2-7b

# Fine-tune
tao model fine-tune \
    --model llama-2-7b \
    --data /data/train.jsonl \
    --output_dir /output \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5
```

### TAO Workflow

```python
# 1. Download pre-trained model
tao model download --model llama-2-7b --output /models

# 2. Prepare dataset (JSONL format)
# {"text": "instruction: ... response: ..."}

# 3. Fine-tune with LoRA
tao model fine-tune \
    --model /models/llama-2-7b \
    --data /data/train.jsonl \
    --peft lora \
    --lora_rank 16 \
    --output /output/finetuned

# 4. Evaluate
tao model evaluate \
    --model /output/finetuned \
    --data /data/eval.jsonl

# 5. Export for deployment
tao model export \
    --model /output/finetuned \
    --format tensorrt \
    --output /export
```

### TAO Supported Operations

| Operation | Description |
|-----------|-------------|
| Fine-tuning | Adapt models to specific domains |
| LoRA/QLoRA | Parameter-efficient fine-tuning |
| Pruning | Remove unnecessary weights |
| Quantization | INT8/FP8 optimization |
| Distillation | Create smaller models |
| Export | TensorRT, ONNX formats |

---

## Base Command Platform

Enterprise cluster management for AI workloads.

### Features

- **Job Scheduling**: Submit and manage training jobs
- **Multi-tenancy**: Resource allocation across teams
- **Dataset Management**: Version and share datasets
- **Experiment Tracking**: Monitor training runs
- **Model Registry**: Store and version models

### Base Command CLI

```bash
# Login
ngc base-command login

# List clusters
ngc base-command cluster list

# Submit training job
ngc base-command job run \
    --name "llama-finetune" \
    --team my_team \
    --ace my_ace \
    --instance dgxa100.80g.8.norm \
    --image nvcr.io/nvidia/nemo:24.05 \
    --commandline "python train.py" \
    --result /results \
    --datasetid 12345:/data

# Monitor job
ngc base-command job info <job_id>

# View logs
ngc base-command job log <job_id>

# Cancel job
ngc base-command job kill <job_id>
```

### Instance Types

| Instance | GPUs | GPU Memory | Use Case |
|----------|------|------------|----------|
| dgxa100.80g.1.norm | 1×A100 | 80 GB | Development |
| dgxa100.80g.8.norm | 8×A100 | 640 GB | Training |
| dgxh100.80g.8.norm | 8×H100 | 640 GB | Large-scale training |

### Dataset Management

```bash
# Upload dataset
ngc dataset upload --source /local/data --destination my_dataset

# List datasets
ngc dataset list

# Mount in job
ngc base-command job run \
    --datasetid my_dataset:/data \
    ...
```

---

## NVIDIA AI Enterprise

Enterprise software platform for production AI.

### Components

| Component | Purpose |
|-----------|---------|
| NeMo | LLM training and customization |
| NIM | Optimized inference microservices |
| Triton | Model serving |
| RAPIDS | GPU-accelerated data processing |
| NeMo Guardrails | Safety and compliance |
| TAO | Transfer learning |

### AI Enterprise Features

- **Enterprise Support**: 24/7 support, SLAs
- **Security**: CVE patching, hardened containers
- **Certified Containers**: Tested and validated
- **Long-term Support**: Extended container lifecycle

### Licensing

```
NVIDIA AI Enterprise Tiers:
1. AI Enterprise Essentials - NIM, limited support
2. AI Enterprise - Full platform
3. AI Enterprise Global - Multi-region deployment
```

### Deployment Options

| Deployment | Description |
|------------|-------------|
| On-premises | DGX, certified servers |
| Cloud | AWS, Azure, GCP, OCI |
| Hybrid | Mix of on-prem and cloud |
| DGX Cloud | Fully managed NVIDIA cloud |

---

## DGX Cloud

Fully managed AI supercomputing platform.

### Features

```
DGX Cloud:
- Instant access to DGX SuperPOD
- No hardware management
- Pre-configured with NeMo, PyTorch
- Integrated with NGC
- Pay-as-you-go pricing
```

### Usage

```bash
# Access via Base Command
ngc base-command cluster list

# Submit job to DGX Cloud
ngc base-command job run \
    --ace nv-us-west-2 \
    --instance dgxh100.80g.8.norm \
    --replicas 4 \
    --image nvcr.io/nvidia/nemo:24.05 \
    --commandline "torchrun train.py"
```

### DGX Cloud Specifications

| Resource | Specification |
|----------|---------------|
| GPU | H100 80GB |
| Interconnect | NVLink, InfiniBand |
| Storage | High-performance NFS |
| Networking | Multi-tenant isolated |

---

## NeMo Curator

Large-scale data curation for LLM training.

### Features

- GPU-accelerated data processing
- Deduplication (exact and fuzzy)
- Quality filtering
- PII redaction
- Language detection

### Usage

```python
from nemo_curator import (
    Sequential,
    DocumentFilter,
    MinHashDeduplicator,
    PIIModifier,
    LanguageFilter
)

# Define curation pipeline
pipeline = Sequential([
    # Filter by quality
    DocumentFilter(
        filter_fn=quality_filter,
        filter_field='text'
    ),

    # Filter by language
    LanguageFilter(
        languages=['en'],
        threshold=0.8
    ),

    # Deduplicate
    MinHashDeduplicator(
        jaccard_threshold=0.8,
        num_hashes=128
    ),

    # Remove PII
    PIIModifier(
        supported_entities=['EMAIL', 'PHONE', 'PERSON'],
        anonymize_action='replace'
    )
])

# Process data
curated_data = pipeline(raw_data)
```

### Scaling with Dask

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# Create GPU cluster
cluster = LocalCUDACluster(n_workers=8)
client = Client(cluster)

# Process at scale
from nemo_curator import distributed_process
result = distributed_process(pipeline, data_path="s3://bucket/data/")
```

---

## Tool Selection Guide

### When to Use What

| Need | Tool |
|------|------|
| Quick LLM deployment | NIM |
| Custom LLM training | NeMo |
| High-performance inference | TensorRT-LLM + Triton |
| Transfer learning | TAO |
| Enterprise deployment | AI Enterprise |
| Cluster management | Base Command |
| Model/container access | NGC |
| Data curation | NeMo Curator |
| GPU data processing | RAPIDS |

### Integration Patterns

```
Development Workflow:
1. Pull containers from NGC
2. Fine-tune with NeMo or TAO
3. Curate data with NeMo Curator
4. Convert with TensorRT-LLM
5. Deploy with Triton or NIM
6. Add safety with NeMo Guardrails
7. Monitor with DCGM

Production Workflow:
1. Use AI Enterprise certified containers
2. Deploy on Base Command or DGX Cloud
3. Manage with NGC
4. Scale with Kubernetes + GPU Operator
```

---

## Quick Reference

### Environment Setup

```bash
# Essential environment variables
export NGC_API_KEY=your_api_key
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Docker with GPU
docker run --gpus all \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v /data:/data \
    nvcr.io/nvidia/nemo:24.05
```

### Common Commands

```bash
# NGC
ngc registry image list                    # List images
ngc registry model list                    # List models
docker pull nvcr.io/nvidia/nemo:24.05     # Pull container

# Base Command
ngc base-command job list                  # List jobs
ngc base-command job run ...              # Submit job
ngc base-command job info <id>            # Job status

# TAO
tao model list                             # List models
tao model fine-tune ...                   # Fine-tune
tao model export ...                      # Export

# NIM
docker run --gpus all nvcr.io/nim/...     # Run NIM
curl localhost:8000/v1/chat/completions   # API call
```

---

## Troubleshooting

| Issue | Symptoms | Solution |
|-------|----------|----------|
| NGC login fails | 401 error | Check API key, regenerate if needed |
| Container pull slow | Timeout | Use regional mirror, check network |
| Job stuck pending | No resources | Check quota, use smaller instance |
| TAO OOM | CUDA OOM | Reduce batch size, enable gradient checkpointing |
| NIM slow startup | Long initialization | Pre-pull image, use local cache |
| Base Command timeout | Job killed | Increase time limit, checkpoint |
