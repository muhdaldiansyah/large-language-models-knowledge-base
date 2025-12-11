# 5.1 Machine Learning Fundamentals

## Overview

Core ML concepts that underpin Large Language Models.

```
Data → Features → Model → Training → Evaluation → Deployment
                    ↓
            Neural Network
```

---

## Neural Network Basics

### Perceptron (Single Neuron)

```
Inputs (x₁, x₂, ..., xₙ) → Weighted Sum → Activation → Output
                               ↓
                    z = Σ(wᵢxᵢ) + b
                               ↓
                        y = σ(z)
```

**Components:**
- **Weights (w):** Learnable parameters
- **Bias (b):** Offset term
- **Activation (σ):** Non-linear function

### Multi-Layer Perceptron (MLP)

```
Input Layer → Hidden Layer(s) → Output Layer
   (x)           (h)              (y)

h = activation(W₁x + b₁)
y = activation(W₂h + b₂)
```

### Forward Propagation

```python
def forward(x, weights, biases):
    """Forward pass through network"""
    a = x
    for W, b in zip(weights, biases):
        z = np.dot(a, W) + b
        a = activation(z)
    return a
```

### Backpropagation

Calculate gradients using chain rule:

```
Loss = L(y, ŷ)

∂L/∂W = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂W
```

```python
def backward(loss, activations, weights):
    """Backward pass to compute gradients"""
    gradients = []
    delta = loss_gradient  # ∂L/∂ŷ

    for layer in reversed(range(num_layers)):
        # Gradient for weights
        grad_W = np.dot(activations[layer].T, delta)
        gradients.append(grad_W)

        # Propagate to previous layer
        delta = np.dot(delta, weights[layer].T) * activation_derivative

    return gradients[::-1]
```

### Gradient Descent

Update weights to minimize loss:

```
W = W - α × ∂L/∂W

α = learning rate
```

**Variants:**

| Optimizer | Update Rule | Use Case |
|-----------|-------------|----------|
| SGD | W -= α × g | Simple, baseline |
| Momentum | v = βv + g; W -= αv | Faster convergence |
| Adam | Adaptive learning rates | Most LLM training |
| AdamW | Adam + weight decay | Recommended for transformers |

```python
# AdamW (commonly used for LLMs)
def adamw_step(params, grads, m, v, t, lr=1e-4, beta1=0.9, beta2=0.999, wd=0.01):
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads**2

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    params = params - lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + wd * params)
    return params, m, v
```

---

## Learning Paradigms

### Supervised Learning

Learn from labeled data: input-output pairs.

```
Training: (x₁, y₁), (x₂, y₂), ... → Model → f(x) ≈ y
```

**Examples:**
- Classification: Spam detection
- Regression: Price prediction
- Sequence-to-sequence: Translation

**LLM connection:** Pre-training uses next-token prediction (supervised on sequences)

### Unsupervised Learning

Learn patterns without labels.

```
Training: x₁, x₂, x₃, ... → Model → Discover structure
```

**Examples:**
- Clustering: Group similar items
- Dimensionality reduction: PCA, t-SNE
- Density estimation: Learn data distribution

**LLM connection:** Understanding text structure, embeddings

### Self-Supervised Learning

Create labels from data itself.

```
Input: "The cat sat on the [MASK]"
Label: "mat" (derived from original text)
```

**Examples:**
- Masked language modeling (BERT)
- Next token prediction (GPT)
- Contrastive learning (embeddings)

**This is how LLMs are trained!**

### Reinforcement Learning

Learn from rewards/feedback.

```
Agent → Action → Environment → Reward → Update Policy
  ↑___________________________________|
```

**LLM connection:**
- RLHF (Reinforcement Learning from Human Feedback)
- Reward model training
- Policy optimization (PPO)

---

## Classification vs Regression

### Classification

Predict discrete categories.

```python
# Binary classification
def binary_classify(x, threshold=0.5):
    prob = sigmoid(model(x))
    return 1 if prob > threshold else 0

# Multi-class classification
def multiclass_classify(x):
    logits = model(x)
    probs = softmax(logits)
    return argmax(probs)
```

**Loss functions:**
- Binary: Binary Cross-Entropy
- Multi-class: Categorical Cross-Entropy

```python
# Cross-entropy loss
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))
```

### Regression

Predict continuous values.

```python
def regression(x):
    return model(x)  # Direct output, no activation
```

**Loss functions:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- Huber Loss (robust to outliers)

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
```

### Comparison

| Aspect | Classification | Regression |
|--------|----------------|------------|
| Output | Categories | Continuous |
| Final activation | Softmax/Sigmoid | None/Linear |
| Loss | Cross-entropy | MSE/MAE |
| Metrics | Accuracy, F1 | RMSE, R² |

---

## Named Entity Recognition (NER)

Identify and classify named entities in text.

```
Input: "Apple CEO Tim Cook announced the new iPhone in Cupertino."

Output:
- "Apple" → ORG (Organization)
- "Tim Cook" → PER (Person)
- "iPhone" → PRODUCT
- "Cupertino" → LOC (Location)
```

### BIO Tagging Scheme

```
Token:  Apple  CEO   Tim   Cook  announced  the  new  iPhone  in  Cupertino
Tag:    B-ORG  O     B-PER I-PER O          O    O    B-PROD  O   B-LOC

B = Beginning of entity
I = Inside entity (continuation)
O = Outside (not an entity)
```

### NER as Sequence Labeling

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def extract_entities(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    entities = []
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    for token, pred in zip(tokens, predictions[0]):
        label = model.config.id2label[pred.item()]
        if label != "O":
            entities.append((token, label))

    return entities
```

### Common Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| PER | Person | "Elon Musk", "Marie Curie" |
| ORG | Organization | "Google", "United Nations" |
| LOC | Location | "Paris", "Mount Everest" |
| DATE | Date/Time | "January 2024", "last week" |
| MONEY | Currency | "$500", "50 euros" |
| PRODUCT | Product | "iPhone", "Windows 11" |

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

Dimensionality reduction for visualization.

```
High-dimensional data (e.g., 768D embeddings)
                ↓
        t-SNE algorithm
                ↓
2D or 3D visualization
```

### How t-SNE Works

1. **Compute pairwise similarities** in high-dimensional space
2. **Initialize** random low-dimensional points
3. **Optimize** low-dimensional positions to match similarities

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce 768D embeddings to 2D
embeddings = get_embeddings(texts)  # Shape: (n_samples, 768)

tsne = TSNE(
    n_components=2,
    perplexity=30,      # Balance local/global structure
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

embeddings_2d = tsne.fit_transform(embeddings)

# Visualize
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
plt.title("t-SNE Visualization of Text Embeddings")
plt.show()
```

### Key Parameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| perplexity | Local vs global structure | 5-50 |
| learning_rate | Optimization speed | 10-1000 |
| n_iter | Convergence | 250-1000 |

### t-SNE vs PCA

| Aspect | t-SNE | PCA |
|--------|-------|-----|
| Type | Non-linear | Linear |
| Preserves | Local structure | Global variance |
| Speed | Slow | Fast |
| Use case | Visualization | Dimensionality reduction |
| Reproducibility | Stochastic | Deterministic |

### Use Cases in LLMs

- Visualize embedding clusters
- Explore semantic similarity
- Debug fine-tuning effects
- Analyze token representations

---

## Activation Functions

### Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

Range: (0, 1)
Use: Binary classification, gates
```

### Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Range: (-1, 1)
Use: Hidden layers (older networks)
```

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)

Range: [0, ∞)
Use: Most hidden layers
Problem: "Dying ReLU" (neurons stuck at 0)
```

### Leaky ReLU

```
LeakyReLU(x) = max(αx, x)  where α = 0.01

Fixes dying ReLU problem
```

### GELU (Gaussian Error Linear Unit)

```
GELU(x) = x × Φ(x)  where Φ = CDF of normal distribution

Use: Transformers (BERT, GPT)
```

### Softmax

```
softmax(xᵢ) = e^(xᵢ) / Σⱼe^(xⱼ)

Output: Probability distribution (sums to 1)
Use: Multi-class classification output
```

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)
```

---

## Loss Functions

### Cross-Entropy Loss

For classification (LLM next-token prediction).

```python
# Binary
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Multi-class
def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))
```

### Mean Squared Error

For regression tasks.

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
```

### Contrastive Loss

For embedding learning.

```python
def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    """InfoNCE loss for contrastive learning"""
    pos_sim = cosine_similarity(anchor, positive) / temperature
    neg_sims = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    logits = [pos_sim] + neg_sims
    labels = 0  # Positive is at index 0

    return cross_entropy_loss(softmax(logits), labels)
```

---

## Regularization

### Dropout

Randomly zero out neurons during training.

```python
def dropout(x, p=0.1, training=True):
    if not training:
        return x
    mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
    return x * mask
```

**LLM usage:** Applied in attention and FFN layers

### Weight Decay (L2 Regularization)

Penalize large weights.

```
Loss_total = Loss_original + λ × Σ(w²)
```

**AdamW** decouples weight decay from gradient update.

### Layer Normalization

Normalize across features (used in transformers).

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

---

## Key Concepts Summary

| Concept | Description | LLM Relevance |
|---------|-------------|---------------|
| Backpropagation | Gradient computation | Core training algorithm |
| Adam/AdamW | Optimizer | Standard for LLM training |
| Self-supervised | Labels from data | How LLMs learn |
| Cross-entropy | Classification loss | Next-token prediction |
| Dropout | Regularization | Prevents overfitting |
| Layer Norm | Normalization | Transformer stability |
| Softmax | Probability distribution | Token selection |
| Embeddings | Dense representations | Input to LLMs |

---

## Common Pitfalls

1. **Vanishing gradients:** Use ReLU/GELU, residual connections
2. **Exploding gradients:** Gradient clipping, proper initialization
3. **Overfitting:** Dropout, weight decay, more data
4. **Underfitting:** Larger model, longer training
5. **Learning rate issues:** Use warmup, schedulers
