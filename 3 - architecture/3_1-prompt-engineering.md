# 3.1 Prompt Engineering (13%)

## Overview

Prompt engineering is the art of crafting inputs to get optimal outputs from LLMs.

```
[System Prompt] + [User Prompt] + [Examples] → LLM → [Response]
```

---

## Prompting Techniques

### Zero-Shot

No examples provided. Model relies on pre-trained knowledge.

```
Prompt: "Translate to French: Hello, how are you?"
Output: "Bonjour, comment allez-vous?"
```

**Best for:**
- Simple, well-defined tasks
- Common operations the model has seen during training
- When you don't have examples

### One-Shot

Single example provided.

```
Prompt:
"Translate to French:
English: Good morning
French: Bonjour

English: Hello, how are you?
French:"

Output: "Bonjour, comment allez-vous?"
```

### Few-Shot

Multiple examples (typically 3-5) to establish pattern.

```
Prompt:
"Classify the sentiment:

Text: I love this product!
Sentiment: Positive

Text: This is terrible.
Sentiment: Negative

Text: It's okay, nothing special.
Sentiment: Neutral

Text: Best purchase I've ever made!
Sentiment:"

Output: "Positive"
```

**Best practices:**
- Use diverse, representative examples
- Keep examples consistent in format
- Order can matter (recent examples weighted more)

### Chain-of-Thought (CoT)

Encourage step-by-step reasoning.

```
Prompt:
"Solve this step by step:
If a train travels 120 miles in 2 hours, then stops for 30 minutes,
then travels another 90 miles in 1.5 hours, what is the average speed
for the entire journey including the stop?"

Output:
"Let me solve this step by step:
1. Total distance = 120 + 90 = 210 miles
2. Travel time = 2 + 1.5 = 3.5 hours
3. Stop time = 0.5 hours
4. Total time = 3.5 + 0.5 = 4 hours
5. Average speed = 210 / 4 = 52.5 mph

The average speed is 52.5 mph."
```

**Trigger phrases:**
- "Let's think step by step"
- "Solve this step by step"
- "Walk me through your reasoning"

### Zero-Shot CoT

Add reasoning trigger without examples.

```
Prompt: "What is 17 × 24? Let's think step by step."

Output:
"Let's think step by step:
17 × 24 = 17 × (20 + 4)
= (17 × 20) + (17 × 4)
= 340 + 68
= 408"
```

### System Instructions

Set behavior, persona, and constraints.

```
System: "You are a helpful coding assistant. Always provide code examples
in Python. Explain your code with comments. If you're unsure about
something, say so rather than guessing."

User: "How do I read a CSV file?"
```

---

## Output Control Parameters

### Temperature

Controls randomness/creativity (0.0 - 2.0).

```
temperature = 0.0  → Deterministic, most likely tokens
temperature = 0.7  → Balanced (good default)
temperature = 1.0  → More creative/varied
temperature = 2.0  → Very random (often incoherent)
```

**Use cases:**
| Temperature | Use Case |
|-------------|----------|
| 0.0 - 0.3 | Code generation, factual Q&A, classification |
| 0.5 - 0.7 | General conversation, explanations |
| 0.8 - 1.0 | Creative writing, brainstorming |

### Top-p (Nucleus Sampling)

Sample from tokens whose cumulative probability ≤ p.

```
top_p = 0.1  → Only most likely tokens (conservative)
top_p = 0.9  → Wider selection (default)
top_p = 1.0  → Consider all tokens
```

**How it works:**
```
Token probabilities: [0.4, 0.3, 0.15, 0.1, 0.05]
top_p = 0.7 → Include tokens until sum ≥ 0.7
Selected: [0.4, 0.3] (sum = 0.7)
Only these tokens are candidates for sampling
```

### Top-k

Only consider the k most likely tokens.

```
top_k = 1   → Greedy decoding (always pick most likely)
top_k = 10  → Choose from top 10 tokens
top_k = 50  → Choose from top 50 tokens (default)
```

### Combining Parameters

```python
# Focused, factual output
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    top_p=0.9,
)

# Creative output
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.9,
    top_p=0.95,
)
```

### Stop Sequences

Halt generation at specific strings.

```python
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "List 3 colors:"}],
    stop=["\n4.", "---", "END"]  # Stop at any of these
)
```

### Max Tokens

Limit output length.

```python
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100  # Maximum output tokens
)
```

---

## Structured Outputs

### JSON Mode

Force valid JSON output.

```python
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{
        "role": "user",
        "content": "Extract name and age from: John is 25 years old."
    }],
    response_format={"type": "json_object"}
)

# Output: {"name": "John", "age": 25}
```

### JSON Schema

Enforce specific structure.

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_schema",
        "json_schema": schema
    }
)
```

### Constrained Generation

Use grammar or regex constraints.

```python
# Only allow specific format
response = generate(
    prompt="Generate a date:",
    regex=r"\d{4}-\d{2}-\d{2}"  # YYYY-MM-DD
)
```

---

## Prompt Templates

### Classification Template

```
Classify the following text into one of these categories: {categories}

Text: {input_text}

Category:
```

### Extraction Template

```
Extract the following information from the text:
- {field1}
- {field2}
- {field3}

Text: {input_text}

Output in JSON format:
```

### Summarization Template

```
Summarize the following text in {num_sentences} sentences.
Focus on the main points and key takeaways.

Text: {input_text}

Summary:
```

### Code Generation Template

```
Write a {language} function that {task_description}.

Requirements:
- {requirement1}
- {requirement2}

Include comments explaining the code.
```

---

## Advanced Techniques

### Self-Consistency

Generate multiple responses and take majority vote.

```python
responses = []
for _ in range(5):
    response = generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

final_answer = most_common(responses)
```

### Prompt Chaining

Break complex tasks into steps.

```
Step 1: Summarize the document
Step 2: Extract key entities from summary
Step 3: Generate questions about entities
Step 4: Answer questions using document
```

### ReAct (Reasoning + Acting)

Interleave thinking with tool use.

```
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first.
Action: Search[capital of France]
Observation: Paris is the capital of France.

Thought: Now I need to find the population of Paris.
Action: Search[population of Paris]
Observation: The population of Paris is about 2.1 million.

Thought: I have the answer.
Answer: The population of Paris, the capital of France, is about 2.1 million.
```

---

## Best Practices

1. **Be specific:** Vague prompts get vague responses
2. **Provide context:** Background info improves accuracy
3. **Use examples:** Few-shot often beats zero-shot
4. **Iterate:** Test and refine prompts
5. **Control output:** Use structured formats when needed
6. **Set temperature appropriately:** Low for facts, higher for creativity
7. **Use system prompts:** Define behavior and constraints upfront