# Context Retention Assistant

A Python-based system for improving context retention and coherence in long-form conversations with language models. The system uses advanced NLP techniques including BERT-based semantic similarity scoring and ROUGE-based lexical overlap analysis to maintain conversation coherence and factual consistency.

## Features

- **Prompt Tuning**: 

- **Context Management**: 


- **Coherence Metrics**: 
  - BERT-based semantic similarity scoring
  - ROUGE-based lexical overlap measurement
  - Weighted ensemble scoring approach
  - Temporal coherence degradation tracking


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

