# 3.3 Retrieval-Augmented Generation (RAG)

## Overview

RAG combines retrieval systems with generative models to provide accurate, up-to-date, and grounded responses.

```
Query → Retriever → Relevant Documents → LLM → Grounded Response
                         ↓
              Vector DB (Milvus, FAISS, Pinecone)
```

**Why RAG?**
- Reduces hallucinations with factual grounding
- Enables access to private/proprietary data
- No need to fine-tune for domain knowledge
- Easy to update (just update the document store)

---

## RAG Pipeline Stages

### 1. Ingestion

Loading and preparing source documents.

```python
# Example document loading
from langchain.document_loaders import PDFLoader, TextLoader

loader = PDFLoader("document.pdf")
documents = loader.load()
```

**Supported formats:**
- PDF, Word, HTML
- Markdown, Plain text
- Code files
- Structured data (CSV, JSON)

### 2. Chunking

Splitting documents into smaller pieces for embedding.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

**Chunking Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| Fixed size | Split at character count | Simple, predictable |
| Recursive | Try separators in order | General text |
| Sentence | Split at sentence boundaries | Narrative text |
| Semantic | Split by topic/meaning | Complex documents |
| Code | Respect code structure | Source code |

**Chunk Size Considerations:**
```
Too small: Loses context, incomplete information
Too large: Dilutes relevance, exceeds model context
Sweet spot: 256-1024 tokens (depends on use case)
```

**Overlap:**
- Prevents information loss at boundaries
- Typical: 10-20% of chunk size

### 3. Embedding

Convert text chunks to dense vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# NVIDIA embedding example
import requests

response = requests.post(
    "https://integrate.api.nvidia.com/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "nvidia/nv-embedqa-e5-v5",
        "input": chunks,
        "input_type": "passage"
    }
)
```

**NVIDIA Embedding Models:**

| Model | Dimensions | Use Case |
|-------|------------|----------|
| NV-Embed-QA | 1024 | Question-answering |
| NV-EmbedQA-E5-v5 | 1024 | Enterprise QA |
| NV-Embed-v2 | 4096 | High accuracy |

**Embedding Best Practices:**
- Use same model for queries and documents
- Normalize vectors for cosine similarity
- Consider instruction-tuned embeddings for queries

### 4. Indexing

Store embeddings in vector database for efficient retrieval.

```python
# FAISS example
import faiss

dimension = 1024
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
index.add(embeddings)

# Milvus example
from pymilvus import Collection, FieldSchema, CollectionSchema

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields)
collection = Collection("documents", schema)
collection.insert([ids, embeddings, texts])
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP"})
```

**Vector Database Comparison:**

| Database | Type | Strengths |
|----------|------|-----------|
| FAISS | Library | Fast, GPU support, Facebook |
| Milvus | Distributed | Scalable, cloud-native |
| Pinecone | Managed | Easy setup, serverless |
| Weaviate | Hybrid | Built-in ML, GraphQL |
| Chroma | Embedded | Simple, local-first |
| Qdrant | Self-hosted | Filtering, payloads |

### 5. Retrieval

Find relevant documents for a query using Approximate Nearest Neighbor (ANN).

```python
# Encode query
query_embedding = model.encode(["What is quantization?"])

# Search
distances, indices = index.search(query_embedding, k=5)
relevant_docs = [documents[i] for i in indices[0]]
```

**ANN Algorithms:**

| Algorithm | Speed | Accuracy | Memory |
|-----------|-------|----------|--------|
| Flat (exact) | Slow | 100% | High |
| IVF | Fast | ~95% | Medium |
| HNSW | Very fast | ~98% | High |
| PQ | Very fast | ~90% | Low |

**Retrieval Parameters:**
- **k (top-k):** Number of documents to retrieve
- **score_threshold:** Minimum similarity score
- **filter:** Metadata-based filtering

### 6. Reranking

Re-score retrieved documents for better relevance.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score each query-document pair
pairs = [[query, doc] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by score
reranked = sorted(zip(scores, retrieved_docs), reverse=True)
```

**Why Rerank?**
- Bi-encoders (embeddings): fast but approximate
- Cross-encoders (rerankers): slow but accurate
- Use bi-encoder for retrieval, cross-encoder for reranking

**NVIDIA Reranking:**
```python
response = requests.post(
    "https://integrate.api.nvidia.com/v1/ranking",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": "nvidia/nv-rerankqa-mistral-4b-v3",
        "query": {"text": query},
        "passages": [{"text": doc} for doc in documents]
    }
)
```

### 7. Generation

Combine retrieved context with query for the LLM.

```python
context = "\n\n".join([doc.text for doc in reranked_docs[:3]])

prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}

Answer:"""

response = llm.generate(prompt)
```

---

## Advanced RAG Patterns

### Hybrid Search

Combine dense (semantic) and sparse (keyword) retrieval.

```python
# BM25 for keyword search
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(tokenized_query)

# Combine with vector search
final_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
```

**Benefits:**
- Captures exact matches (BM25)
- Captures semantic similarity (vectors)
- More robust retrieval

### Multi-Query RAG

Generate multiple query variations to improve retrieval.

```python
# Generate query variations
variations = llm.generate(f"""
Generate 3 different versions of this question:
{original_query}
""")

# Retrieve for each variation
all_docs = set()
for query in variations:
    docs = retrieve(query)
    all_docs.update(docs)
```

### Self-Query RAG

Extract structured filters from natural language.

```
User: "Find documents about transformers from 2023"
        ↓
Query: "transformers"
Filter: {"year": 2023}
```

### Parent Document Retrieval

Retrieve small chunks but return larger parent context.

```
Document → Large chunks (parents) → Small chunks (children)
                                          ↓
                                    Embed & retrieve
                                          ↓
                                    Return parent
```

### HyDE (Hypothetical Document Embeddings)

Generate hypothetical answer, use it for retrieval.

```python
# Generate hypothetical answer
hypothetical = llm.generate(f"Write a passage that answers: {query}")

# Use hypothetical for retrieval (often better than query embedding)
docs = retrieve(hypothetical)
```

---

## RAG Evaluation Metrics

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Recall@k | % of relevant docs in top k |
| Precision@k | % of top k that are relevant |
| MRR | Mean Reciprocal Rank |
| NDCG | Normalized Discounted Cumulative Gain |

### Generation Metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Is answer supported by context? |
| Answer Relevance | Does answer address the question? |
| Context Relevance | Is retrieved context useful? |

### RAG Evaluation Frameworks

- **RAGAS:** Open-source RAG evaluation
- **TruLens:** Feedback functions for RAG
- **LangSmith:** LangChain's evaluation platform

---

## Function Calling / Tool Use

LLMs can call external functions/APIs when needed.

### OpenAI-style Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Model returns function call
# {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}
```

### Tool Use Pattern

```
User Query → LLM decides tool needed → Call tool → LLM processes result → Response
                    ↓
            Returns structured call
```

---

## NVIDIA RAG Components

### NeMo Retriever

NVIDIA's enterprise RAG service:
- NV-Embed models for embedding
- NV-Rerank models for reranking
- Optimized for NVIDIA GPUs

### NVIDIA AI Foundation Endpoints

```python
# Embedding endpoint
nvidia_embed = "https://integrate.api.nvidia.com/v1/embeddings"

# Reranking endpoint
nvidia_rerank = "https://integrate.api.nvidia.com/v1/ranking"

# Generation endpoint
nvidia_llm = "https://integrate.api.nvidia.com/v1/chat/completions"
```

### Integration with Milvus

```python
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Use NVIDIA embeddings with Milvus
collection = Collection("nvidia_rag")
collection.search(
    data=[nvidia_embedding],
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=10
)
```

---

## RAG Best Practices

1. **Chunking:** Match chunk size to your content type
2. **Overlap:** Use 10-20% overlap to preserve context
3. **Metadata:** Store and filter by metadata (date, source, type)
4. **Reranking:** Always rerank for better precision
5. **Evaluation:** Measure retrieval AND generation quality
6. **Hybrid:** Combine semantic and keyword search
7. **Prompt:** Instruct LLM to use only provided context
8. **Citations:** Have LLM cite sources for traceability
