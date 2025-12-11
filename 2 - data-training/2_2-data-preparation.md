# 2.2 Data Preparation (9%)

## Tokenization

Tokenization converts text into numerical tokens that models can process.

```
"Hello world" → [15496, 995] → Model → [Output tokens] → "Response"
```

---

## Tokenization Algorithms

### BPE (Byte Pair Encoding)

Most common algorithm for modern LLMs (GPT, LLaMA, etc.)

**How it works:**
1. Start with character-level vocabulary
2. Count all adjacent pairs
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

```
Training example:
Corpus: "low lower lowest"

Step 1: Characters: l, o, w, e, r, s, t, (space)
Step 2: Most frequent pair: "l" + "o" → "lo"
Step 3: Most frequent pair: "lo" + "w" → "low"
Step 4: Continue until vocab_size reached

Result vocabulary: [..., "low", "er", "est", ...]
```

**Tokenization example:**
```
Input: "lowest"
Tokens: ["low", "est"]
IDs: [1234, 5678]
```

### WordPiece

Used by BERT and similar models.

**Differences from BPE:**
- Uses likelihood-based scoring instead of frequency
- Adds `##` prefix for subword continuations

```
Input: "unbelievable"
Tokens: ["un", "##believ", "##able"]
```

**Scoring formula:**
```
score(pair) = freq(pair) / (freq(first) × freq(second))
```

### SentencePiece

Language-agnostic tokenizer (doesn't assume spaces = word boundaries).

**Key features:**
- Treats input as raw byte stream
- Works for any language (Chinese, Japanese, etc.)
- Includes BPE and Unigram algorithms
- Used by T5, LLaMA, Mistral

```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe'  # or 'unigram'
)

# Use tokenizer
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode("Hello world", out_type=str)
# ['▁Hello', '▁world']
```

**Special token:** `▁` (U+2581) represents space/word boundary

---

## Vocabulary Size Impact

| Vocab Size | Pros | Cons |
|------------|------|------|
| Small (8K) | Smaller embedding table, faster | More tokens per text, longer sequences |
| Medium (32K) | Good balance | Standard choice |
| Large (100K+) | Fewer tokens, shorter sequences | Large embedding table, rare token issues |

**Embedding table size:**
```
Memory = vocab_size × hidden_dim × bytes_per_param
Example: 32,000 × 4,096 × 2 (FP16) = 256 MB
```

**Sequence length impact:**
```
Text: "The quick brown fox"
- Small vocab: 6 tokens (more subwords)
- Large vocab: 4 tokens (whole words)
```

---

## Special Tokens

Common special tokens in LLM vocabularies:

| Token | Purpose |
|-------|---------|
| `<bos>` / `<s>` | Beginning of sequence |
| `<eos>` / `</s>` | End of sequence |
| `<pad>` | Padding for batch alignment |
| `<unk>` | Unknown/OOV token |
| `<mask>` | Masked token (for MLM) |
| `[INST]` / `[/INST]` | Instruction delimiters |
| `<<SYS>>` | System prompt markers |

---

## NeMo Data Processing

### Data Formats

**Binary format (.bin + .idx):**
- Efficient for large-scale training
- Memory-mapped for fast access
- Created by NeMo preprocessing scripts

```bash
# Convert JSONL to binary format
python preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed/train \
    --tokenizer-model tokenizer.model \
    --workers 32
```

**Output files:**
```
processed/train_document.bin  # Token IDs
processed/train_document.idx  # Index/offsets
```

### Pipeline Configuration

```yaml
# NeMo data config
data:
  data_prefix:
    - 0.7  # weight
    - /data/train_document  # path prefix
    - 0.3
    - /data/code_document

  index_mapping_dir: /cache/index_mapping

  splits_string: "98,1,1"  # train, val, test

  seq_length: 4096

  skip_warmup: false
```

---

## RAPIDS Ecosystem

GPU-accelerated data processing libraries.

### cuDF (GPU DataFrames)

pandas-like API on GPU.

```python
import cudf

# Load data on GPU
df = cudf.read_parquet("data.parquet")

# Operations run on GPU
df['text_length'] = df['text'].str.len()
filtered = df[df['text_length'] > 100]

# 10-100x faster than pandas for large data
```

### cuML (GPU Machine Learning)

scikit-learn-like API on GPU.

```python
from cuml.cluster import KMeans
from cuml.decomposition import PCA

# Clustering for data analysis
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(embeddings_gpu)

# Dimensionality reduction
pca = PCA(n_components=50)
reduced = pca.fit_transform(embeddings_gpu)
```

### cuGraph (GPU Graph Analytics)

Graph algorithms on GPU.

```python
import cugraph

# Build graph from edges
G = cugraph.Graph()
G.from_cudf_edgelist(edges_df, source='src', destination='dst')

# Run PageRank
pagerank = cugraph.pagerank(G)

# Community detection
partitions = cugraph.louvain(G)
```

---

## Data Pipeline Architecture

```
Raw Data (text files, web crawl)
         ↓
    [Extraction]
         ↓
    [Filtering] ← Quality heuristics, language detection
         ↓
    [Deduplication] ← Exact hash, MinHash LSH
         ↓
    [PII Removal] ← Regex, NER-based detection
         ↓
    [Tokenization] ← BPE/SentencePiece
         ↓
    [Packing] ← Combine short docs to seq_length
         ↓
Binary Files (.bin/.idx) → Training
```

---

## Deduplication

Removing duplicate or near-duplicate content is critical for training quality.

### Exact Deduplication (Hash-based)

```python
import hashlib
from collections import defaultdict

def exact_dedup(documents):
    """Remove exact duplicates using MD5 hash."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        # Normalize and hash
        normalized = doc.strip().lower()
        doc_hash = hashlib.md5(normalized.encode()).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs
```

### Fuzzy Deduplication (MinHash LSH)

For near-duplicates with minor variations.

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    """Create MinHash signature for a document."""
    mh = MinHash(num_perm=num_perm)
    # Create shingles (n-grams)
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        mh.update(shingle.encode('utf-8'))
    return mh

def fuzzy_dedup(documents, threshold=0.8):
    """Remove near-duplicates using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_docs = []

    for i, doc in enumerate(documents):
        mh = create_minhash(doc)

        # Check for similar documents
        result = lsh.query(mh)
        if not result:
            lsh.insert(f"doc_{i}", mh)
            unique_docs.append(doc)

    return unique_docs
```

### N-gram Deduplication

```python
def ngram_dedup(documents, n=5, threshold=0.9):
    """Remove documents with high n-gram overlap."""
    def get_ngrams(text, n):
        words = text.split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

    unique_docs = []
    seen_ngrams = set()

    for doc in documents:
        doc_ngrams = get_ngrams(doc, n)
        if not doc_ngrams:
            continue

        # Calculate overlap with seen content
        overlap = len(doc_ngrams & seen_ngrams) / len(doc_ngrams)

        if overlap < threshold:
            unique_docs.append(doc)
            seen_ngrams.update(doc_ngrams)

    return unique_docs
```

### Deduplication at Scale (Spark + RAPIDS)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import md5, concat_ws, col

spark = SparkSession.builder \
    .config("spark.rapids.sql.enabled", "true") \
    .getOrCreate()

# Load data
df = spark.read.parquet("documents.parquet")

# Exact dedup
df_deduped = df.dropDuplicates(["text"])

# Hash-based dedup
df = df.withColumn("text_hash", md5(col("text")))
df_deduped = df.dropDuplicates(["text_hash"])
```

---

## Data Contamination Detection

Ensuring test data hasn't leaked into training data.

### N-gram Overlap Detection

```python
def check_contamination(train_texts, test_texts, n=13):
    """Check for n-gram overlap between train and test."""

    # Build n-gram index from training data
    train_ngrams = set()
    for text in train_texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            train_ngrams.add(ngram)

    # Check test data for matches
    contaminated = []
    for i, text in enumerate(test_texts):
        words = text.split()
        matches = 0
        total = len(words) - n + 1

        for j in range(total):
            ngram = ' '.join(words[j:j+n])
            if ngram in train_ngrams:
                matches += 1

        if total > 0 and matches / total > 0.8:
            contaminated.append(i)

    return contaminated
```

### Embedding-based Contamination Check

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def embedding_contamination_check(train_texts, test_texts, threshold=0.95):
    """Find test samples too similar to training data."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    train_embeddings = model.encode(train_texts)
    test_embeddings = model.encode(test_texts)

    contaminated = []
    for i, test_emb in enumerate(test_embeddings):
        # Find max similarity to any training sample
        similarities = np.dot(train_embeddings, test_emb) / (
            np.linalg.norm(train_embeddings, axis=1) * np.linalg.norm(test_emb)
        )
        if similarities.max() > threshold:
            contaminated.append(i)

    return contaminated
```

---

## Quality Filtering

### Heuristic Filters

```python
def quality_filter(text):
    """Apply heuristic quality filters."""
    # Length check
    if len(text) < 100 or len(text) > 100000:
        return False

    # Word count
    words = text.split()
    if len(words) < 20:
        return False

    # Average word length (detect garbage)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 15:
        return False

    # Alphanumeric ratio
    alnum_chars = sum(c.isalnum() for c in text)
    if alnum_chars / len(text) < 0.7:
        return False

    # Repetition check
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.2:
        return False

    # Stop word presence (indicates real text)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
    if not any(w.lower() in stop_words for w in words):
        return False

    return True
```

### Perplexity-based Filtering

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text, model, tokenizer, max_length=512):
    """Calculate perplexity as quality score."""
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss

    return torch.exp(loss).item()

# Filter out high-perplexity (low quality) text
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

filtered_docs = [doc for doc in documents
                 if calculate_perplexity(doc, model, tokenizer) < 100]
```

### Classifier-based Filtering

```python
from transformers import pipeline

# Use a quality classifier
quality_classifier = pipeline(
    "text-classification",
    model="HuggingFaceFW/fineweb-edu-classifier"
)

def classify_quality(texts, threshold=3):
    """Filter by educational/quality score."""
    results = quality_classifier(texts)
    return [
        text for text, result in zip(texts, results)
        if float(result['label'].split('_')[-1]) >= threshold
    ]
```

---

## Language Detection and Filtering

```python
from langdetect import detect, detect_langs
import fasttext

# FastText language detection (more accurate)
model = fasttext.load_model('lid.176.bin')

def detect_language_fasttext(text):
    """Detect language using FastText."""
    predictions = model.predict(text.replace('\n', ' '), k=1)
    lang = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    return lang, confidence

def filter_by_language(documents, target_langs=['en'], min_confidence=0.8):
    """Keep only documents in target languages."""
    filtered = []
    for doc in documents:
        lang, conf = detect_language_fasttext(doc)
        if lang in target_langs and conf >= min_confidence:
            filtered.append(doc)
    return filtered
```

---

## Multilingual Data Processing

### Handling Multiple Scripts

```python
import unicodedata

def normalize_unicode(text):
    """Normalize Unicode for consistent processing."""
    # NFC normalization (composed form)
    text = unicodedata.normalize('NFC', text)

    # Remove control characters
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc')

    return text

def detect_script(text):
    """Detect primary script of text."""
    scripts = {}
    for char in text:
        if char.isalpha():
            script = unicodedata.name(char, '').split()[0]
            scripts[script] = scripts.get(script, 0) + 1

    if scripts:
        return max(scripts, key=scripts.get)
    return 'Unknown'
```

### Parallel Corpus Alignment

```python
def align_parallel_sentences(src_sentences, tgt_sentences, embedder):
    """Align sentences in parallel corpora using embeddings."""
    src_embs = embedder.encode(src_sentences)
    tgt_embs = embedder.encode(tgt_sentences)

    # Compute similarity matrix
    similarities = np.dot(src_embs, tgt_embs.T)

    # Greedy alignment
    alignments = []
    for i in range(len(src_sentences)):
        best_j = similarities[i].argmax()
        if similarities[i, best_j] > 0.8:
            alignments.append((i, best_j, similarities[i, best_j]))

    return alignments
```

---

## PII Detection and Removal

### Regex-based PII Detection

```python
import re

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

def redact_pii(text, replacement='[REDACTED]'):
    """Remove PII using regex patterns."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'{replacement}_{pii_type.upper()}', text)
    return text
```

### NER-based PII Detection (Presidio)

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def anonymize_text(text):
    """Detect and anonymize PII using Presidio."""
    # Analyze for PII
    results = analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                  "CREDIT_CARD", "US_SSN", "LOCATION"],
        language='en'
    )

    # Anonymize detected PII
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text
```

---

## Data Mixing and Sampling

### Domain Weighting

```python
def create_mixed_dataset(datasets, weights, total_samples):
    """Mix datasets according to weights."""
    mixed = []
    samples_per_dataset = [int(w * total_samples) for w in weights]

    for dataset, n_samples in zip(datasets, samples_per_dataset):
        if len(dataset) >= n_samples:
            sampled = random.sample(dataset, n_samples)
        else:
            # Upsample if needed
            sampled = random.choices(dataset, k=n_samples)
        mixed.extend(sampled)

    random.shuffle(mixed)
    return mixed

# Example: Mix code and text data
mixed_data = create_mixed_dataset(
    datasets=[code_data, text_data, math_data],
    weights=[0.3, 0.5, 0.2],
    total_samples=1_000_000
)
```

### Temperature Sampling

```python
def temperature_sample(datasets, temperatures, total_samples):
    """Sample with temperature to adjust distribution."""
    # Calculate effective weights
    sizes = np.array([len(d) for d in datasets])
    weights = sizes ** (1.0 / np.array(temperatures))
    weights = weights / weights.sum()

    return create_mixed_dataset(datasets, weights, total_samples)
```

---

## Synthetic Data Generation

### Using LLMs for Data Augmentation

```python
def generate_synthetic_samples(prompt_template, seed_examples, llm, n_samples):
    """Generate synthetic training data using an LLM."""
    synthetic_data = []

    for _ in range(n_samples):
        # Select random seed example
        seed = random.choice(seed_examples)

        prompt = prompt_template.format(example=seed)
        response = llm.generate(prompt, max_tokens=500)

        synthetic_data.append({
            'input': response['input'],
            'output': response['output'],
            'synthetic': True
        })

    return synthetic_data
```

### Self-Instruct Method

```python
def self_instruct(base_instructions, llm, n_generate=1000):
    """Generate instruction-following data using Self-Instruct."""
    generated = []

    for _ in range(n_generate):
        # Sample seed instructions
        seeds = random.sample(base_instructions, k=3)

        prompt = f"""Generate a new instruction similar to these examples:
1. {seeds[0]}
2. {seeds[1]}
3. {seeds[2]}

New instruction:"""

        new_instruction = llm.generate(prompt)

        # Generate input-output pair
        instance = llm.generate(f"Create an example for: {new_instruction}")

        generated.append({
            'instruction': new_instruction,
            'input': instance.get('input', ''),
            'output': instance['output']
        })

    return generated
```

---

## Best Practices

1. **Tokenizer training:**
   - Train on representative corpus
   - Include all languages/domains you'll use
   - 32K vocab is good default for LLMs

2. **Data quality:**
   - Quality > quantity for fine-tuning
   - Deduplicate aggressively (both exact and fuzzy)
   - Remove PII for compliance
   - Filter low-quality content

3. **Contamination prevention:**
   - Check for train/test overlap
   - Use n-gram and embedding-based detection
   - Document data sources and versions

4. **Efficiency:**
   - Use binary formats for training
   - Pre-tokenize to avoid repeated work
   - Use RAPIDS/Spark for large-scale processing
   - Cache intermediate results

5. **Validation:**
   - Check tokenizer coverage
   - Verify no data leakage between splits
   - Monitor token distribution
   - Sample and manually inspect data

6. **Documentation:**
   - Track data provenance
   - Record filtering decisions
   - Version control data pipelines
   - Log deduplication statistics
