"""
Demonstration of the Context Retention Assistant system.
"""

from src.prompt_tuning.prompt_tuner import PromptTuner
from src.metrics.coherence_metrics import CoherenceMetrics
from src.context_injection.context_manager import ContextManager
from src.evaluation.evaluator import SystemEvaluator

def main():
    # Initialize components
    print("Initializing Context Retention Assistant...")
    prompt_tuner = PromptTuner()
    metrics = CoherenceMetrics()
    context_manager = ContextManager()
    evaluator = SystemEvaluator()
    
    # Sample conversation
    conversation = [
        {"role": "user", "content": "What are the key concepts in deep learning?"},
        {"role": "assistant", "content": "Deep learning involves neural networks with multiple layers that can learn hierarchical representations. Key concepts include backpropagation, activation functions, and gradient descent."},
        {"role": "user", "content": "How does backpropagation work?"}
    ]
    
    # Key facts to maintain
    facts = [
        "Deep learning uses neural networks with multiple layers",
        "Key concepts include backpropagation and gradient descent",
        "Neural networks learn hierarchical representations"
    ]
    
    print("\n1. Testing Prompt Tuning...")
    tuned_prompt = prompt_tuner.tune_prompt(conversation, facts)
    print(f"Enhanced Prompt:\n{tuned_prompt}\n")
    
    print("2. Measuring Coherence...")
    consistency = metrics.measure_factual_consistency(
        facts,
        [conversation[1]["content"]]
    )
    print(f"Factual Consistency Score: {consistency:.2f}\n")
    
    print("3. Testing Context Injection...")
    enhanced_context = context_manager.inject_context(
        conversation[-1]["content"],
        conversation[:-1]
    )
    print(f"Context-Enhanced Prompt:\n{enhanced_context}\n")
    
    print("4. Running System Evaluation...")
    test_conversations = [conversation]
    ground_truth = [{
        "facts": facts,
        "relevant_context": ["deep learning", "neural networks", "backpropagation"]
    }]
    
    metrics = evaluator.evaluate_system(test_conversations, ground_truth)
    print("\nSystem Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()