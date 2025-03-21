# Context Retention Assistant

A Python-based system for improving context retention and coherence in long-form conversations with language models. The system uses advanced NLP techniques including BERT-based semantic similarity scoring and ROUGE-based lexical overlap analysis to maintain conversation coherence and factual consistency.

## Features

- **Prompt Tuning**: 
  - Adaptive prompt enhancement based on conversation history
  - Dynamic memory key points integration
  - Automatic context window management
  - GPU-accelerated when available

- **Context Management**: 
  - Intelligent context selection using semantic similarity
  - Token-aware context compression
  - Adaptive context injection with length validation
  - Configurable context window sizes

- **Coherence Metrics**: 
  - BERT-based semantic similarity scoring
  - ROUGE-based lexical overlap measurement
  - Weighted ensemble scoring approach
  - Temporal coherence degradation tracking

- **Evaluation Framework**: 
  - Comprehensive system evaluation
  - Detailed performance reporting
  - Configurable metric thresholds
  - JSON-formatted analysis output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/context-retention-assistant.git
cd context-retention-assistant
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
context-retention-assistant/
├── src/
│   ├── prompt_tuning/
│   │   └── prompt_tuner.py      # Prompt enhancement and tuning
│   ├── context_injection/
│   │   └── context_manager.py   # Context management system
│   ├── metrics/
│   │   └── coherence_metrics.py # Coherence measurement tools
│   └── evaluation/
│       └── evaluator.py         # System evaluation framework
├── tests/
│   └── test_system.py          # Integration tests
├── requirements.txt            # Project dependencies
└── README.md                  
```

## Usage

### Basic Example

```python
from src.prompt_tuning.prompt_tuner import PromptTuner
from src.context_injection.context_manager import ContextManager
from src.evaluation.evaluator import SystemEvaluator

# Initialize components with GPU support if available
prompt_tuner = PromptTuner(model_name="gpt2")
context_manager = ContextManager(model_name="gpt2", max_length=1024)
evaluator = SystemEvaluator()

# Example conversation
conversation_history = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you explain supervised learning?"}
]

# Enhance prompt with context
memory_points = ["ML is a subset of AI", "Systems learn from experience"]
tuned_prompt = prompt_tuner.tune_prompt(conversation_history, memory_points)

# Inject relevant context
enhanced_prompt = context_manager.inject_context(
    tuned_prompt, 
    conversation_history,
    max_context_length=512
)

# Evaluate system performance
test_conversations = [{
    "id": "test1",
    "history": conversation_history,
    "responses": ["Supervised learning uses labeled data..."]
}]

ground_truth = {
    "test1": [
        "Supervised learning requires labeled data",
        "Models learn to map inputs to outputs"
    ]
}

metrics = evaluator.evaluate_system(test_conversations, ground_truth)
report = evaluator.generate_evaluation_report(metrics)
print(report)
```

### Metrics and Evaluation

The system provides three main metrics for evaluation:

1. **Factual Consistency** (threshold: 0.7)
   - Combines BERT semantic similarity (70%) and ROUGE lexical overlap (30%)
   - Measures how well responses maintain factual accuracy
   - Score range: 0.0 to 1.0 (higher is better)

2. **Coherence Loss** (threshold: 0.3)
   - Tracks degradation in conversation coherence over time
   - Uses sliding window analysis of semantic similarity
   - Score range: 0.0+ (lower is better)

3. **Context Relevance** (threshold: 0.6)
   - Evaluates effectiveness of context injection
   - Measures semantic similarity between responses and selected context
   - Score range: 0.0 to 1.0 (higher is better)

### Advanced Configuration

```python
# Configure PromptTuner with specific device
prompt_tuner = PromptTuner(
    model_name="gpt2",
    device="cuda"  # or "cpu"
)

# Configure ContextManager with custom token limits
context_manager = ContextManager(
    model_name="gpt2",
    max_length=2048  # Increase token limit
)

# Generate detailed evaluation report
report = evaluator.generate_evaluation_report(
    metrics,
    output_path="evaluation_report.json"
)
```

## Testing

Run the test suite:

```bash
python -m unittest tests/test_system.py -v
```

The test suite includes:
- Prompt tuning functionality
- Coherence metrics calculation
- Context injection behavior
- End-to-end system evaluation

## Performance Considerations

- GPU acceleration is automatically used when available
- Memory management includes automatic cleanup
- Token length validation prevents model context overflow
- Configurable thresholds for all evaluation metrics

