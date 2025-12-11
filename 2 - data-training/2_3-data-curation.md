# 2.3 Advanced Data Curation

## Overview

Data curation ensures training data is high-quality, deduplicated, and compliant with privacy requirements.

```
Raw Web Data → Curation Pipeline → Clean Training Data
  (Petabytes)                         (Terabytes)
   ~99% filtered out
```

---

## Scalability with Dask + RAPIDS

### Why Dask?

- Parallelizes operations across multiple GPUs/nodes
- Handles datasets larger than memory
- Familiar pandas/cuDF API

```python
import dask_cudf

# Read partitioned data
ddf = dask_cudf.read_parquet("data/*.parquet")

# Lazy operations (builds task graph)
ddf['clean_text'] = ddf['text'].str.lower()
ddf = ddf[ddf['text'].str.len() > 50]

# Execute on cluster
result = ddf.compute()
```

### Cluster Setup

```python
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

# Create GPU cluster
cluster = LocalCUDACluster(n_workers=8)  # 8 GPUs
client = Client(cluster)

# Process at scale
ddf = dask_cudf.read_parquet("s3://bucket/data/")
result = ddf.map_partitions(process_fn).compute()
```

---

## Deduplication

### Why Deduplicate?

- Repeated data causes memorization
- Inflates perplexity metrics artificially
- Wastes training compute
- Legal/copyright concerns

### Exact Deduplication (Hash-based)

Fast but only catches identical documents.

```python
import hashlib

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# Deduplicate
seen_hashes = set()
unique_docs = []

for doc in documents:
    h = get_hash(doc['text'])
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_docs.append(doc)
```

**Considerations:**
- Normalize text first (lowercase, whitespace)
- Can use n-gram hashing for paragraph-level dedup
- Very fast: O(n) with hash set

### Fuzzy Deduplication (MinHash LSH)

Catches near-duplicates (e.g., same article with different formatting).

**MinHash Algorithm:**
1. Convert document to n-gram set
2. Apply multiple hash functions
3. Keep minimum hash value for each function
4. Similar documents have similar MinHash signatures

```python
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode())
    return m

# Build LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=128)

for i, doc in enumerate(documents):
    mh = get_minhash(doc['text'])
    lsh.insert(f"doc_{i}", mh)

# Query for near-duplicates
query_mh = get_minhash(query_doc)
duplicates = lsh.query(query_mh)
```

**Jaccard Similarity:**
```
J(A, B) = |A ∩ B| / |A ∪ B|

Documents with J > 0.8 typically considered duplicates
```

### GPU-Accelerated Deduplication

```python
# Using NVIDIA NeMo Curator
from nemo_curator import MinHashDeduplicator

deduplicator = MinHashDeduplicator(
    num_hashes=128,
    jaccard_threshold=0.8,
    num_buckets=20
)

# Process on GPU
deduplicated = deduplicator(dataset)
```

---

## Quality Filtering

### Heuristic Filters

Rule-based filtering for obvious issues:

```python
def quality_filter(doc):
    text = doc['text']

    # Length filters
    if len(text) < 100:
        return False
    if len(text) > 100000:
        return False

    # Word count
    words = text.split()
    if len(words) < 20:
        return False

    # Average word length (filters gibberish)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 15:
        return False

    # Symbol ratio
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / len(text) < 0.7:
        return False

    # Repetition detection
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / len(lines) < 0.5:
        return False

    return True
```

### Common Heuristics

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Min length | 100 chars | Remove stubs |
| Max length | 100K chars | Remove dumps |
| Alpha ratio | >70% | Filter code/data |
| Unique lines | >50% | Detect repetition |
| Stop word ratio | 5-30% | Detect gibberish |
| Sentence count | >3 | Ensure coherence |
| Max line length | <1000 | Filter logs/data |

### Classifier-based Filtering

Train a model to predict document quality.

```python
from transformers import AutoModelForSequenceClassification

# Load quality classifier
model = AutoModelForSequenceClassification.from_pretrained(
    "quality-classifier"
)

def classify_quality(texts):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=-1)[:, 1]  # P(high quality)
    return scores

# Filter by score
high_quality = [doc for doc, score in zip(docs, scores) if score > 0.8]
```

**Training a quality classifier:**
1. Sample documents manually labeled as high/low quality
2. Fine-tune BERT/RoBERTa on binary classification
3. Apply to full corpus

### Language Detection

```python
from fasttext import load_model

# FastText language ID
lang_model = load_model('lid.176.bin')

def detect_language(text):
    predictions = lang_model.predict(text.replace('\n', ' '))
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return lang, confidence

# Filter for English
english_docs = [
    doc for doc in docs
    if detect_language(doc['text'])[0] == 'en'
]
```

---

## PII Redaction

### Why Remove PII?

- GDPR, CCPA compliance
- Prevent model memorization of personal data
- Ethical considerations
- Reduce legal risk

### PII Types

| Type | Examples | Detection Method |
|------|----------|-----------------|
| Email | john@email.com | Regex |
| Phone | 555-123-4567 | Regex |
| SSN | 123-45-6789 | Regex |
| Credit Card | 4111-1111-1111-1111 | Regex + Luhn |
| Names | John Smith | NER |
| Addresses | 123 Main St | NER |
| IP Address | 192.168.1.1 | Regex |

### Regex-based Detection

```python
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

def redact_pii(text):
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[{pii_type.upper()}]', text)
    return text

# Example
text = "Contact john@email.com or 555-123-4567"
redacted = redact_pii(text)
# "Contact [EMAIL] or [PHONE]"
```

### NER-based Detection

For names, addresses, organizations:

```python
from transformers import pipeline

# Load NER model
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def redact_entities(text):
    entities = ner(text)

    # Sort by position (reverse to preserve indices)
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)

    for entity in entities:
        if entity['entity_group'] in ['PER', 'LOC', 'ORG']:
            placeholder = f"[{entity['entity_group']}]"
            text = text[:entity['start']] + placeholder + text[:entity['end']]

    return text
```

### GPU-Accelerated PII Detection

```python
# Using NVIDIA NeMo Curator
from nemo_curator.modules import PIIModifier

pii_modifier = PIIModifier(
    supported_entities=['EMAIL', 'PHONE', 'PERSON', 'LOCATION'],
    anonymize_action='replace'  # or 'hash', 'mask'
)

cleaned_dataset = pii_modifier(dataset)
```

---

## Complete Curation Pipeline

```python
from nemo_curator import (
    DocumentFilter,
    MinHashDeduplicator,
    PIIModifier,
    Sequential
)

pipeline = Sequential([
    # Quality filtering
    DocumentFilter(
        filter_fn=quality_filter,
        filter_field='text'
    ),

    # Language filter
    DocumentFilter(
        filter_fn=lambda x: detect_language(x)[0] == 'en'
    ),

    # Fuzzy deduplication
    MinHashDeduplicator(
        jaccard_threshold=0.8,
        num_hashes=128
    ),

    # PII removal
    PIIModifier(
        supported_entities=['EMAIL', 'PHONE', 'PERSON'],
        anonymize_action='replace'
    ),
])

# Run pipeline
clean_data = pipeline(raw_data)
```

---

## Best Practices

1. **Order matters:** Filter before dedup (faster on smaller data)
2. **Log statistics:** Track how much data each stage removes
3. **Sample and verify:** Manually check filtered/kept examples
4. **Iterate:** Adjust thresholds based on downstream performance
5. **Version data:** Track which curation pipeline produced each dataset
6. **Test PII removal:** Validate with known PII examples
