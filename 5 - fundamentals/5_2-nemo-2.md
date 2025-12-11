# 5.2 NeMo 2.0 & Infrastructure

## Overview

NeMo 2.0 is NVIDIA's next-generation framework for building, training, and deploying LLMs at scale.

```
NeMo 2.0 Stack:
┌─────────────────────────────────────────┐
│           NeMo Framework                │
├─────────────────────────────────────────┤
│  NeMo-Run  │  AutoModel  │  Recipes     │
├─────────────────────────────────────────┤
│         Megatron-Core                   │
├─────────────────────────────────────────┤
│  PyTorch  │  CUDA  │  NCCL  │  cuDNN    │
└─────────────────────────────────────────┘
```

---

## NeMo 2.0 Key Changes

### From YAML to Python Config

**NeMo 1.x (YAML-based):**
```yaml
# config.yaml
model:
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32

trainer:
  max_steps: 100000
  precision: bf16

data:
  data_path: /data/train.jsonl
```

**NeMo 2.0 (Python-based):**
```python
from nemo.collections.llm import GPTModel, GPTConfig
from nemo.lightning import Trainer

# Model configuration
config = GPTConfig(
    hidden_size=4096,
    num_layers=32,
    num_attention_heads=32,
    ffn_hidden_size=14336,
    seq_length=4096,
)

# Create model
model = GPTModel(config)

# Trainer
trainer = Trainer(
    max_steps=100000,
    precision="bf16-mixed",
    accelerator="gpu",
    devices=8,
    strategy="ddp"
)

# Train
trainer.fit(model, train_dataloader)
```

### Benefits of Python Config

| Aspect | YAML | Python |
|--------|------|--------|
| Type checking | No | Yes (IDE support) |
| Autocomplete | No | Yes |
| Debugging | Difficult | Easy |
| Composition | Limited | Full Python power |
| Validation | Runtime | Pre-execution |
| Versioning | Text diff | Code review |

---

## NeMo-Run

Execution framework for running NeMo experiments locally or on clusters.

### Basic Usage

```python
import nemo_run as run

# Define experiment
@run.cli.factory
def my_experiment():
    return run.Partial(
        train,
        model=GPTConfig(hidden_size=4096),
        trainer=TrainerConfig(max_steps=10000),
        data=DataConfig(path="/data/train.jsonl")
    )

# Run locally
run.run(my_experiment(), executor=run.LocalExecutor())

# Run on Slurm
run.run(my_experiment(), executor=run.SlurmExecutor(
    nodes=4,
    ntasks_per_node=8,
    gpus_per_task=1,
    partition="batch"
))
```

### Executors

```python
# Local execution
local = run.LocalExecutor()

# Slurm cluster
slurm = run.SlurmExecutor(
    account="project_name",
    partition="gpu",
    nodes=8,
    ntasks_per_node=8,
    time="24:00:00",
    exclusive=True
)

# Kubernetes
k8s = run.KubernetesExecutor(
    namespace="nemo-training",
    image="nvcr.io/nvidia/nemo:24.05",
    num_gpus=8
)
```

### Experiment Tracking

```python
# Configure logging
experiment = run.Experiment(
    name="llama-finetune",
    log_dir="/results/experiments",
    tensorboard=True,
    wandb={"project": "nemo-training"}
)

# Run with tracking
with experiment:
    run.run(training_task, executor=slurm)
```

---

## AutoModel

Simplified interface for loading and using pre-trained models.

### Loading Models

```python
from nemo.collections.llm import AutoModel

# Load from NGC
model = AutoModel.from_pretrained("nvidia/llama-3.1-8b-nemo")

# Load from HuggingFace
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B")

# Load from local checkpoint
model = AutoModel.from_pretrained("/path/to/checkpoint")
```

### Model Operations

```python
# Generate text
output = model.generate(
    "What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

# Get embeddings
embeddings = model.encode(["text1", "text2"])

# Fine-tune
model.finetune(
    train_data="/data/train.jsonl",
    val_data="/data/val.jsonl",
    epochs=3
)
```

### Export for Deployment

```python
# Export to TensorRT-LLM
model.export(
    output_dir="/models/exported",
    format="trt-llm",
    precision="fp8"
)

# Export to ONNX
model.export(
    output_dir="/models/onnx",
    format="onnx"
)
```

---

## Pre-built Recipes

Ready-to-use training configurations for common tasks.

### Fine-tuning Recipe

```python
from nemo.collections.llm.recipes import llama3_8b_finetune

# Get pre-configured recipe
recipe = llama3_8b_finetune(
    data_path="/data/train.jsonl",
    num_gpus=8,
    max_steps=1000
)

# Customize
recipe.trainer.max_steps = 2000
recipe.model.lora.enabled = True
recipe.model.lora.rank = 16

# Run
run.run(recipe, executor=run.SlurmExecutor(nodes=1))
```

### Pre-training Recipe

```python
from nemo.collections.llm.recipes import llama3_8b_pretrain

recipe = llama3_8b_pretrain(
    data_path="/data/corpus",
    tokenizer_path="/tokenizers/llama",
    num_nodes=16,
    num_gpus_per_node=8
)

# Configure parallelism
recipe.parallelism.tensor_parallel = 2
recipe.parallelism.pipeline_parallel = 4
recipe.parallelism.data_parallel = 16

run.run(recipe, executor=slurm_executor)
```

### Available Recipes

| Recipe | Use Case |
|--------|----------|
| `llama3_8b_finetune` | Fine-tune Llama 3.1 8B |
| `llama3_70b_pretrain` | Pre-train Llama 3.1 70B |
| `mistral_7b_finetune` | Fine-tune Mistral 7B |
| `nemotron_4_340b` | NVIDIA Nemotron training |
| `mixtral_8x7b` | MoE model training |

---

## Infrastructure: Slurm

Job scheduler for HPC clusters.

### Slurm Basics

```bash
# Submit job
sbatch train.sh

# Check queue
squeue -u $USER

# Cancel job
scancel <job_id>

# Interactive session
srun --nodes=1 --gpus=8 --time=1:00:00 --pty bash
```

### Job Script

```bash
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=project_name
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load modules
module load cuda/12.2
module load nccl/2.18

# Set environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Run training
srun python train.py \
    --nodes=$SLURM_NNODES \
    --gpus=8 \
    --config=config.yaml
```

### Multi-Node Training

```python
# NeMo-Run handles Slurm automatically
executor = run.SlurmExecutor(
    account="my_project",
    partition="gpu",
    nodes=4,
    ntasks_per_node=8,
    gpus_per_task=1,
    time="48:00:00",
    container_image="nvcr.io/nvidia/nemo:24.05",
    env_vars={
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"
    }
)
```

---

## Infrastructure: Kubernetes

Container orchestration for cloud deployments.

### NeMo on Kubernetes

```yaml
# nemo-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-finetune
spec:
  template:
    spec:
      containers:
      - name: nemo
        image: nvcr.io/nvidia/nemo:24.05
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 8
        volumeMounts:
        - name: data
          mountPath: /data
        - name: results
          mountPath: /results
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data
      - name: results
        persistentVolumeClaim:
          claimName: training-results
      restartPolicy: Never
```

### NVIDIA GPU Operator

```yaml
# Enable GPU scheduling
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: cuda-container
    image: nvcr.io/nvidia/cuda:12.2-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

### Multi-Node Training on K8s

```yaml
# Using PyTorch Operator
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: nemo-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/nemo:24.05
            resources:
              limits:
                nvidia.com/gpu: 8
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/nemo:24.05
            resources:
              limits:
                nvidia.com/gpu: 8
```

---

## DGX SuperPOD

NVIDIA's reference architecture for large-scale AI training.

### Architecture

```
DGX SuperPOD Components:
┌─────────────────────────────────────────────────────┐
│                   Management                         │
│         (Slurm, Kubernetes, Monitoring)             │
├─────────────────────────────────────────────────────┤
│                 InfiniBand Fabric                    │
│              (400 Gb/s per port)                    │
├─────────────────────────────────────────────────────┤
│  DGX H100  │  DGX H100  │  DGX H100  │  DGX H100   │
│  8×H100    │  8×H100    │  8×H100    │  8×H100     │
│  NVLink    │  NVLink    │  NVLink    │  NVLink     │
└─────────────────────────────────────────────────────┘
```

### DGX H100 Specifications

| Component | Specification |
|-----------|---------------|
| GPUs | 8× NVIDIA H100 80GB |
| GPU Memory | 640 GB HBM3 |
| GPU Interconnect | NVLink 4.0 (900 GB/s) |
| Network | 8× 400 Gb/s InfiniBand |
| System Memory | 2 TB |
| Storage | 30 TB NVMe SSD |

### SuperPOD Scalability

```
Single DGX:     8 GPUs
DGX Pod:        256 GPUs (32 DGX)
SuperPOD:       1024+ GPUs (128+ DGX)
```

### Network Topology

```
         ┌─────────────────┐
         │   Spine Switch  │
         └────────┬────────┘
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐    ┌───────┐    ┌───────┐
│ Leaf  │    │ Leaf  │    │ Leaf  │
└───┬───┘    └───┬───┘    └───┬───┘
    │            │            │
┌───┴───┐   ┌───┴───┐   ┌───┴───┐
│DGX x4 │   │DGX x4 │   │DGX x4 │
└───────┘   └───────┘   └───────┘
```

---

## Base Command Platform

NVIDIA's cluster management software.

### Features

- Job scheduling and queuing
- Multi-tenancy and resource allocation
- Dataset management
- Experiment tracking
- Model registry

### Usage

```bash
# Submit training job
ngc base-command job run \
    --name "llama-finetune" \
    --image "nvcr.io/nvidia/nemo:24.05" \
    --instance "dgxa100.80g.8.norm" \
    --commandline "python train.py" \
    --result /results

# List jobs
ngc base-command job list

# Monitor job
ngc base-command job info <job_id>
```

---

## NeMo 2.0 Migration Guide

### Key Differences

| NeMo 1.x | NeMo 2.0 |
|----------|----------|
| YAML config | Python config |
| OmegaConf | Pydantic |
| PTL Trainer | NeMo Trainer |
| Hydra CLI | NeMo-Run |
| `.nemo` checkpoints | Standard PyTorch |

### Migration Steps

1. **Convert configs:** YAML → Python dataclasses
2. **Update imports:** `nemo.collections.nlp` → `nemo.collections.llm`
3. **Replace trainer:** PTL Trainer → NeMo Trainer
4. **Update checkpoints:** Use conversion utilities
5. **Update CLI:** Hydra → NeMo-Run

### Checkpoint Conversion

```python
from nemo.collections.llm import convert_checkpoint

# NeMo 1.x to 2.0
convert_checkpoint(
    source="/checkpoints/nemo1/model.nemo",
    target="/checkpoints/nemo2/",
    source_format="nemo1",
    target_format="nemo2"
)
```

---

## Best Practices

### Configuration Management

```python
# Use dataclasses for type safety
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    model: GPTConfig
    trainer: TrainerConfig
    data: DataConfig

    def validate(self):
        assert self.model.hidden_size % self.model.num_attention_heads == 0
        assert self.trainer.max_steps > 0
```

### Reproducibility

```python
import nemo_run as run

# Pin versions
run.set_seed(42)

# Log configuration
experiment = run.Experiment(
    name="experiment-v1",
    config=config,
    git_commit=run.get_git_commit()
)
```

### Resource Efficiency

```python
# Compute optimal parallelism
def compute_parallelism(num_gpus, model_size):
    if model_size < 7e9:  # < 7B
        return {"tp": 1, "pp": 1, "dp": num_gpus}
    elif model_size < 70e9:  # < 70B
        return {"tp": 8, "pp": 1, "dp": num_gpus // 8}
    else:  # 70B+
        return {"tp": 8, "pp": 4, "dp": num_gpus // 32}
```
