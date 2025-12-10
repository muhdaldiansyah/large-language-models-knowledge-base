# 1.4 NVIDIA NIM

## Definition

NVIDIA NIM (NVIDIA Inference Microservices) is a set of optimized inference microservices that enable deployment of AI models with minimal setup.

- Pre-optimized containers
- Production-ready inference
- Built on TensorRT-LLM and Triton
- Part of NVIDIA AI Enterprise

---

## Day-1 Deployment

### Quick Start
```bash
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### Key Features
- No model optimization required
- Automatic GPU detection
- Pre-configured for optimal performance
- Supports multiple model architectures

### Supported Models
- Llama 3.1 (8B, 70B, 405B)
- Mistral / Mixtral
- Gemma
- Custom fine-tuned models

---

## OpenAI-compatible API

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### Completions
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "prompt": "The capital of France is",
    "max_tokens": 50
  }'
```

### API Endpoints
- `/v1/chat/completions` - Chat format
- `/v1/completions` - Text completion
- `/v1/models` - List available models
- `/health` - Health check

### Parameters
- `temperature` - Randomness (0-2)
- `top_p` - Nucleus sampling
- `max_tokens` - Output limit
- `stream` - Streaming responses
- `stop` - Stop sequences
