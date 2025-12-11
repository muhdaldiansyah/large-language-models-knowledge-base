# Group E - Fundamentals

## E1. Machine Learning Fundamentals

### Neural Network Basics
- Perceptron and MLP
- Forward/backward propagation
- Gradient descent and optimizers (SGD, Adam, AdamW)
- Loss functions (Cross-entropy, MSE)

### Learning Paradigms
- Supervised learning
- Unsupervised learning
- Self-supervised learning
- Reinforcement learning (RLHF)

### Classification vs Regression
- Output types and activations
- Loss function selection
- Evaluation metrics

### Named Entity Recognition (NER)
- BIO tagging scheme
- Entity types (PER, ORG, LOC, etc.)
- Sequence labeling approach

### Visualization
- t-SNE for embeddings
- PCA vs t-SNE comparison
- Perplexity parameter tuning

### Activation Functions
- Sigmoid, Tanh
- ReLU, Leaky ReLU
- GELU, SiLU (Swish)
- Softmax

### Regularization
- Dropout
- Weight decay (L2)
- Layer normalization

**Detailed notes:** [5_1-ml-fundamentals.md](5_1-ml-fundamentals.md)

---

## E2. NeMo 2.0 & Infrastructure

### NeMo 2.0 Key Changes
- Python config (replacing YAML)
- Pydantic models (replacing OmegaConf)
- Type-safe configuration
- IDE support and autocomplete

### NeMo-Run
- Experiment execution framework
- Local and cluster execution
- Slurm and Kubernetes executors
- Experiment tracking

### AutoModel
- Simplified model loading
- HuggingFace/NGC integration
- Fine-tuning interface
- Export to TensorRT-LLM

### Pre-built Recipes
- Fine-tuning recipes
- Pre-training recipes
- Parallelism configuration

**Detailed notes:** [5_2-nemo-2.md](5_2-nemo-2.md)

---

## E3. Infrastructure

### Slurm
- Job scheduling (sbatch, srun)
- Multi-node training setup
- Resource allocation
- Job scripts

### Kubernetes
- GPU scheduling
- NVIDIA GPU Operator
- PyTorch Operator for distributed training
- Pod and Job configuration

### DGX SuperPOD
- DGX H100 specifications
- NVLink and InfiniBand fabric
- Scalability (Pod → SuperPOD)
- Network topology

### Base Command Platform
- Job submission
- Resource management
- Experiment tracking
