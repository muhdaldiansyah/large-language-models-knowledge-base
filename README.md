# Large Language Models Knowledge Base

A comprehensive reference for production LLM systems using the NVIDIA ecosystem.

## Topics

### 1 - Infrastructure

**1.1. Model Optimization**
- **Quantization Methods** - Techniques to reduce model precision (FP8, INT8, INT4) for faster inference with minimal accuracy loss.
- **TensorRT-LLM Build Flags** - Configuration options for compiling optimized LLM inference engines.
- **AMAX Calibration** - Process of determining optimal scaling factors for quantized models using representative data.
- **KV-Cache Quantization** - Compressing key-value cache memory to enable longer context lengths.
- **In-flight Batching** - Dynamic request grouping during inference to maximize GPU utilization.
- **PagedAttention** - Memory management technique that eliminates fragmentation in KV cache allocation.
- **PTQ vs QAT** - Post-training quantization (fast, no retraining) versus quantization-aware training (better accuracy).
- **Mixed Precision (FP16/BF16)** - Using lower precision formats during computation while maintaining model accuracy.
- **AMP Optimization Levels** - Automatic mixed precision settings (O0, O1, O2) balancing speed and accuracy.
- **Compute Estimation** - Formulas for calculating FLOPs, memory requirements, and throughput metrics.

**1.2. GPU Acceleration**
- **Tensor Parallelism (TP)** - Splitting model layers across GPUs for large models that exceed single GPU memory.
- **Pipeline Parallelism (PP)** - Distributing model layers sequentially across GPUs for very deep networks.
- **Data Parallelism (DP)** - Replicating the model across GPUs and splitting batches for increased throughput.
- **Sequence Parallelism (SP)** - Splitting the sequence dimension across GPUs to reduce activation memory.
- **NeMo 2.0 Configuration** - Python-based configuration system replacing YAML for model training and inference.
- **Batch Configuration** - Settings for micro/global batch sizes and gradient accumulation steps.
- **Nsight Metrics** - GPU profiling measurements including SM occupancy, kernel latency, and memory bandwidth.
- **Memory Calculations** - Formulas for estimating model, optimizer, activation, and KV cache memory requirements.
- **NVLink vs PCIe** - High-speed GPU interconnects (900 GB/s) versus standard CPU-GPU connections (64 GB/s).
- **NCCL** - NVIDIA's collective communications library for multi-GPU and multi-node synchronization.
- **Tensor Cores** - Specialized hardware units for accelerated matrix operations in FP16, BF16, and INT8.
- **DGX Architecture** - NVIDIA's integrated AI infrastructure combining multiple GPUs with high-speed interconnects.

**1.3. Deployment**
- **Triton Inference Server** - NVIDIA's production inference platform supporting multiple frameworks and dynamic batching.
- **Model Repository Structure** - Directory organization for serving versioned models with configuration files.
- **config.pbtxt** - Protocol buffer configuration file defining model inputs, outputs, and serving parameters.
- **Dynamic Batching** - Automatic grouping of incoming requests to improve throughput and GPU utilization.
- **Ensembling** - Chaining multiple models (preprocessing → inference → postprocessing) in a single pipeline.
- **Kubernetes Orchestration** - Container management with horizontal scaling and GPU resource scheduling.
- **Docker / NGC Containers** - Pre-optimized NVIDIA containers for deploying AI workloads.
- **Slurm** - HPC job scheduler for managing GPU resources in cluster environments.
- **CUDA Graph Capture** - Recording kernel launches to reduce CPU overhead for repeated operations.
- **Layer Fusion** - Combining multiple operations to reduce memory bandwidth and improve performance.

**1.4. NVIDIA NIM**
- **NVIDIA NIM** - Pre-optimized inference microservices enabling production deployment with minimal setup.
- **Day-1 Deployment** - Single-command container launch with automatic GPU detection and optimization.
- **OpenAI-compatible API** - Standard REST endpoints (/v1/chat/completions, /v1/completions) for easy integration.
- **Supported Models** - Pre-built containers for Llama 3.1, Mistral, Mixtral, Gemma, and custom fine-tuned models.

### 2 - Data & Training

**2.1. Fine-Tuning**
- **Full Fine-Tuning** - Updating all model parameters on new data for maximum adaptation.
- **LoRA** - Low-rank adaptation that trains small adapter matrices instead of full weights.
- **QLoRA** - Quantized LoRA combining 4-bit base models with trainable adapters for memory efficiency.
- **P-Tuning** - Prompt-based tuning that learns continuous prompt embeddings.
- **SFT** - Supervised fine-tuning on instruction-response pairs to align model behavior.
- **RLHF** - Reinforcement learning from human feedback to optimize for human preferences.
- **DPO** - Direct preference optimization as a simpler alternative to RLHF.

**2.2. Data Preparation**
- **Tokenization (BPE)** - Byte pair encoding that iteratively merges frequent character pairs into tokens.
- **WordPiece** - Subword tokenization that maximizes likelihood of training data.
- **SentencePiece** - Language-agnostic tokenizer treating text as raw bytes.
- **RAPIDS Ecosystem** - GPU-accelerated data processing libraries (cuDF, cuML, cuGraph) for large-scale data.

**2.3. Data Curation**
- **Exact Deduplication** - Hash-based removal of identical documents from training data.
- **Fuzzy Deduplication** - MinHash LSH for finding and removing near-duplicate content.
- **Quality Filtering** - Heuristic and classifier-based approaches to remove low-quality text.
- **PII Redaction** - Detection and masking of personally identifiable information in datasets.

### 3 - Architecture & Applications

**3.1. Prompt Engineering**
- **Zero-shot** - Prompting without examples, relying on model's pre-trained knowledge.
- **Few-shot** - Including example input-output pairs to guide model behavior.
- **Chain-of-Thought** - Prompting the model to show reasoning steps before the final answer.
- **Temperature** - Parameter controlling randomness in output (0 = deterministic, higher = more random).
- **Top-p / Top-k** - Sampling strategies that limit token selection to most probable candidates.
- **Structured Outputs** - Constraining model responses to valid JSON or specific formats.

**3.2. LLM Architecture**
- **Self-Attention** - Mechanism allowing each token to attend to all other tokens in the sequence.
- **Multi-Head Attention** - Running multiple attention operations in parallel for richer representations.
- **Positional Encoding** - Adding position information since transformers have no inherent sequence order.
- **Feed-Forward Network** - Dense layers applied independently to each position after attention.
- **Decoder-only Architecture** - Autoregressive models (GPT-style) that predict the next token.

**3.3. RAG (Retrieval-Augmented Generation)**
- **Embeddings** - Dense vector representations of text for semantic similarity search.
- **Vector Databases** - Specialized storage (Milvus, FAISS, Pinecone) for efficient similarity search.
- **Chunking** - Splitting documents into smaller segments for embedding and retrieval.
- **Reranking** - Using cross-encoders to re-score retrieved documents for relevance.
- **NV-Embed** - NVIDIA's embedding models optimized for retrieval tasks.

**3.4. Architectural Nuances**
- **RoPE** - Rotary position embedding that encodes position through rotation matrices.
- **FlashAttention** - Memory-efficient attention algorithm that reduces memory from O(n²) to O(n).
- **Megatron-Core** - NVIDIA's library for efficient large-scale transformer training.

### 4 - Operations

**4.1. Evaluation**
- **Perplexity** - Measures how well the model predicts text (lower is better).
- **BLEU** - Compares generated text to references using n-gram overlap.
- **ROUGE** - Evaluates summaries by measuring recall of reference n-grams.
- **Exact Match** - Binary metric checking if prediction exactly matches the target.
- **Benchmarking** - Standardized evaluation on datasets like MMLU, HellaSwag, and HumanEval.

**4.2. Production Monitoring**
- **Latency Metrics** - Tracking p50/p95/p99 response times for performance SLAs.
- **Throughput** - Measuring tokens per second or requests per second capacity.
- **GPU Utilization** - Monitoring compute and memory usage for efficiency optimization.
- **Health Checks** - Automated endpoint monitoring for service availability.

**4.3. Safety & Compliance**
- **NeMo Guardrails** - Framework for adding programmable safety controls to LLM applications.
- **Input/Output Rails** - Filters that validate and sanitize requests and responses.
- **Jailbreak Detection** - Identifying attempts to bypass model safety guidelines.
- **Hallucination Detection** - Checking generated content against retrieved facts.
- **PII Masking** - Automatically redacting sensitive information from outputs.
- **Bias Auditing** - Systematic evaluation of model fairness across demographic groups.

**4.4. NVIDIA Tools**
- **NeMo** - End-to-end framework for training, fine-tuning, and deploying LLMs.
- **TensorRT-LLM** - High-performance inference library with quantization and optimization.
- **Triton** - Production inference server supporting multiple models and frameworks.
- **Nsight Systems** - GPU profiling tool for identifying performance bottlenecks.
- **ModelOpt** - Toolkit for model compression, quantization, and optimization.
- **NGC Catalog** - Repository of pre-trained models, containers, and resources.

### 5 - Fundamentals

**5.1. ML Basics**
- **Neural Networks** - Layered computational graphs that learn patterns from data.
- **Supervised Learning** - Training with labeled input-output pairs.
- **NER** - Named entity recognition for extracting structured information from text.
- **t-SNE** - Dimensionality reduction technique for visualizing high-dimensional embeddings.

**5.2. Infrastructure**
- **NeMo 2.0** - Latest NeMo version with Python configs, NeMo-Run, and AutoModel APIs.
- **Slurm** - Workload manager for scheduling jobs on HPC clusters.
- **Kubernetes** - Container orchestration platform for scalable deployments.
- **DGX SuperPOD** - NVIDIA's large-scale AI infrastructure combining multiple DGX systems.

## Key Technologies

NeMo | TensorRT-LLM | Triton | RAPIDS | Milvus | FAISS

## Resources

- [NVIDIA NeMo](https://developer.nvidia.com/nemo)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Triton Inference Server](https://github.com/triton-inference-server)
