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

## Best Practices

1. **Tokenizer training:**
   - Train on representative corpus
   - Include all languages/domains you'll use
   - 32K vocab is good default

2. **Data quality:**
   - Quality > quantity for fine-tuning
   - Deduplicate aggressively
   - Remove PII for compliance

3. **Efficiency:**
   - Use binary formats for training
   - Pre-tokenize to avoid repeated work
   - Use RAPIDS for large-scale processing

4. **Validation:**
   - Check tokenizer coverage
   - Verify no data leakage between splits
   - Monitor token distribution
