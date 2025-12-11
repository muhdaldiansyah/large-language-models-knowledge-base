# 4.1 LLM Evaluation (7%)

## Overview

Evaluation measures how well an LLM performs on specific tasks. Critical for model selection, fine-tuning validation, and production monitoring.

```
Model Output → Metrics → Score → Decision
                  ↓
         Compare to baseline/requirements
```

---

## Core Evaluation Metrics

### Perplexity

Measures how "surprised" the model is by the text. Lower is better.

```
Perplexity = exp(-1/N × Σ log P(token_i | context))
```

**Interpretation:**
- Perplexity = 1: Perfect prediction
- Perplexity = 10: Model chooses between ~10 equally likely tokens
- Perplexity = 100: High uncertainty

**Use cases:**
- Compare language models on same dataset
- Monitor training progress
- Detect distribution shift

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return torch.exp(outputs.loss).item()
```

**Limitations:**
- Only measures next-token prediction
- Doesn't measure task performance
- Can't compare models with different vocabularies

---

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap between generated and reference text.

```
BLEU = BP × exp(Σ w_n × log p_n)

BP = Brevity Penalty (penalizes short outputs)
p_n = Precision of n-grams
w_n = Weight for each n-gram (usually 1/4 each)
```

**Score interpretation:**
| BLEU Score | Quality |
|------------|---------|
| < 10 | Almost useless |
| 10-19 | Hard to understand |
| 20-29 | Clear gist |
| 30-40 | Understandable |
| 40-50 | High quality |
| 50-60 | Very high quality |
| > 60 | Near human |

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']

score = sentence_bleu(reference, candidate)
# Returns ~0.67
```

**Use cases:**
- Machine translation
- Text generation comparison
- Summarization (with caution)

**Limitations:**
- Exact match only (synonyms don't count)
- Doesn't measure fluency or meaning
- Multiple valid outputs penalized

---

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures overlap focusing on recall. Multiple variants.

**ROUGE-N:** N-gram recall
```
ROUGE-N = (Matching n-grams) / (Total n-grams in reference)
```

**ROUGE-L:** Longest Common Subsequence
```
ROUGE-L = LCS(candidate, reference) / length(reference)
```

**ROUGE-Lsum:** ROUGE-L on sentence level, then averaged

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

reference = "The cat sat on the mat."
candidate = "The cat is on the mat."

scores = scorer.score(reference, candidate)
# {'rouge1': Score(precision=0.83, recall=1.0, fmeasure=0.91),
#  'rouge2': Score(precision=0.6, recall=0.75, fmeasure=0.67),
#  'rougeL': Score(precision=0.83, recall=1.0, fmeasure=0.91)}
```

**Use cases:**
- Summarization (primary metric)
- Text generation
- Question answering

---

### Exact Match (EM)

Binary: does the output exactly match the reference?

```python
def exact_match(prediction, reference):
    return prediction.strip().lower() == reference.strip().lower()
```

**Use cases:**
- Question answering (extractive)
- Named entity recognition
- Classification

---

### F1 Score

Harmonic mean of precision and recall at token level.

```
Precision = (Common tokens) / (Tokens in prediction)
Recall = (Common tokens) / (Tokens in reference)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

```python
def token_f1(prediction, reference):
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    common = pred_tokens & ref_tokens

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)
```

**Use cases:**
- Question answering (more lenient than EM)
- Information extraction
- Any task with partial credit

---

### Accuracy

Percentage of correct predictions for classification tasks.

```
Accuracy = Correct predictions / Total predictions
```

**Use cases:**
- Classification tasks
- Multiple choice QA
- Sentiment analysis

---

## Task-Specific Metrics

### Summarization
- ROUGE-1, ROUGE-2, ROUGE-L (primary)
- BERTScore (semantic similarity)
- Factual consistency

### Translation
- BLEU (primary)
- chrF (character-level F-score)
- COMET (learned metric)

### Question Answering
- Exact Match
- F1 Score
- Answer correctness (for generative)

### Code Generation
- Pass@k (% tests passed)
- CodeBLEU
- Execution accuracy

### Dialog
- Perplexity
- Human evaluation
- Task success rate

---

## Evaluation Techniques

### Cross-Validation

Split data into k folds, train on k-1, test on 1, rotate.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)

scores = []
for train_idx, test_idx in kf.split(data):
    model = train(data[train_idx])
    score = evaluate(model, data[test_idx])
    scores.append(score)

final_score = np.mean(scores)
```

**Benefits:**
- More robust estimate
- Uses all data for testing
- Detects overfitting

### Train/Val/Test Split

```
Full Dataset → Train (70%) + Validation (15%) + Test (15%)
                  ↓              ↓                ↓
              Training     Hyperparameter    Final evaluation
                           tuning
```

**Important:** Never tune on test set!

### A/B Testing

Compare models in production with real users.

```
Users → Random Assignment → Model A → Collect metrics
                         → Model B → Collect metrics
                                          ↓
                                    Statistical test
```

**Metrics to track:**
- Task success rate
- User satisfaction
- Engagement metrics
- Error rates

### Error Analysis

Systematic examination of model failures.

```python
# Categorize errors
error_types = {
    'factual': [],
    'formatting': [],
    'incomplete': [],
    'hallucination': [],
    'wrong_task': []
}

for pred, ref in zip(predictions, references):
    if not is_correct(pred, ref):
        error_type = categorize_error(pred, ref)
        error_types[error_type].append((pred, ref))
```

**Error analysis steps:**
1. Sample failed predictions
2. Categorize failure modes
3. Quantify each category
4. Prioritize fixes by impact

---

## LLM Benchmarks

### General Language Understanding

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| MMLU | Multi-task multiple choice | Accuracy |
| HellaSwag | Commonsense reasoning | Accuracy |
| ARC | Science QA | Accuracy |
| WinoGrande | Coreference resolution | Accuracy |

### Reasoning

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| GSM8K | Grade school math | Accuracy |
| MATH | Competition math | Accuracy |
| BBH | Big Bench Hard | Accuracy |
| HumanEval | Code generation | Pass@k |

### Safety

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| TruthfulQA | Truthfulness | % Truthful |
| ToxiGen | Toxicity | % Non-toxic |
| RealToxicity | Toxic completions | Toxicity score |

### Running Benchmarks

```python
# Using lm-evaluation-harness
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu", "hellaswag", "arc_easy"],
    batch_size=8
)

print(results['results'])
```

---

## Semantic Similarity Metrics

### BERTScore

Uses BERT embeddings to measure semantic similarity.

```python
from bert_score import score

P, R, F1 = score(candidates, references, lang='en')
```

**Benefits:**
- Captures synonyms and paraphrases
- More aligned with human judgment
- Works across languages

### BLEURT

Learned metric trained on human judgments.

```python
from bleurt import score as bleurt_score

scorer = bleurt_score.BleurtScorer("BLEURT-20")
scores = scorer.score(references, candidates)
```

---

## Human Evaluation

### Rating Scales

```
Rate the response quality (1-5):
1 - Completely wrong/unusable
2 - Mostly wrong with some correct parts
3 - Partially correct but incomplete
4 - Mostly correct with minor issues
5 - Fully correct and complete
```

### Pairwise Comparison

```
Which response is better?
[ ] Response A
[ ] Response B
[ ] Tie
```

**Benefits:**
- Easier for humans than absolute ratings
- More consistent results
- Can use Elo/TrueSkill rankings

### Key Dimensions to Evaluate

| Dimension | Question |
|-----------|----------|
| Accuracy | Is the information correct? |
| Relevance | Does it answer the question? |
| Completeness | Is anything missing? |
| Fluency | Is it well-written? |
| Helpfulness | Is it useful to the user? |
| Safety | Is it harmful or inappropriate? |

---

## LLM-as-Judge

Use an LLM to evaluate other LLMs.

```python
evaluation_prompt = """
Rate the following response on a scale of 1-5 for accuracy.

Question: {question}
Response: {response}
Reference: {reference}

Provide your rating and explanation.
Rating:"""

judge_response = llm.generate(evaluation_prompt)
```

**Best practices:**
- Use strong model as judge (GPT-4, Claude)
- Provide clear rubrics
- Include reference answers when possible
- Check for position bias (swap order)
- Validate against human judgments

---

## Evaluation Pipeline

```python
class EvaluationPipeline:
    def __init__(self, model, metrics):
        self.model = model
        self.metrics = metrics

    def evaluate(self, test_data):
        results = {'predictions': [], 'scores': {}}

        # Generate predictions
        for item in test_data:
            pred = self.model.generate(item['input'])
            results['predictions'].append(pred)

        # Calculate metrics
        for metric_name, metric_fn in self.metrics.items():
            scores = []
            for pred, item in zip(results['predictions'], test_data):
                score = metric_fn(pred, item['reference'])
                scores.append(score)
            results['scores'][metric_name] = np.mean(scores)

        return results

# Usage
pipeline = EvaluationPipeline(
    model=my_model,
    metrics={
        'bleu': calculate_bleu,
        'rouge': calculate_rouge,
        'f1': token_f1
    }
)
results = pipeline.evaluate(test_data)
```

---

## Best Practices

1. **Multiple metrics:** No single metric captures everything
2. **Test set integrity:** Never tune on test data
3. **Statistical significance:** Report confidence intervals
4. **Human correlation:** Validate automatic metrics
5. **Error analysis:** Understand failure modes
6. **Benchmark diversity:** Test multiple capabilities
7. **Production metrics:** Track real-world performance
8. **Version tracking:** Track evaluation results over time
