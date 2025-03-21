"""
Evaluation framework for testing context retention and coherence.
"""

from typing import List, Dict, Any
import json
from datetime import datetime
from bert_score import score
import numpy as np

from src.prompt_tuning.prompt_tuner import PromptTuner
from src.metrics.coherence_metrics import CoherenceMetrics
from src.context_injection.context_manager import ContextManager

class SystemEvaluator:
    def __init__(self):
        """Initialize evaluation components."""
        self.coherence_metrics = CoherenceMetrics()
        self.prompt_tuner = PromptTuner()
        self.context_manager = ContextManager()
        
    def evaluate_system(self,
                       test_conversations: List[Dict[str, Any]],
                       ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate system performance on test conversations.
        
        Args:
            test_conversations: List of test conversation scenarios
            ground_truth: Dictionary mapping conversation IDs to ground truth facts
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {
            'factual_consistency': [],
            'coherence_loss': [],
            'context_relevance': []
        }
        
        for conv in test_conversations:
            conv_id = conv['id']
            conversation_history = conv['history']
            generated_responses = conv['responses']
            
            # Evaluate factual consistency
            if conv_id in ground_truth:
                consistency = self.coherence_metrics.measure_factual_consistency(
                    ground_truth[conv_id],
                    generated_responses
                )
                metrics['factual_consistency'].append(consistency)
            
            # Evaluate coherence
            coherence_loss = self.coherence_metrics.measure_coherence_loss(
                conversation_history
            )
            metrics['coherence_loss'].append(coherence_loss)
            
            # Evaluate context relevance
            context_score = self._evaluate_context_relevance(
                conversation_history,
                generated_responses
            )
            metrics['context_relevance'].append(context_score)
            
        # Calculate average metrics
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in metrics.items()
        }
        
    def _evaluate_context_relevance(self,
                                  conversation_history: List[Dict[str, str]],
                                  generated_responses: List[str]) -> float:
        """
        Evaluate relevance of injected context.
        
        Args:
            conversation_history: List of conversation turns
            generated_responses: List of model responses
            
        Returns:
            Context relevance score between 0 and 1
        """
        if not conversation_history or not generated_responses:
            return 0.0
            
        relevance_scores = []
        for response in generated_responses:
            # Get context that would have been injected
            relevant_context = self.context_manager._select_relevant_context(
                response,
                conversation_history
            )
            
            if relevant_context:
                # Measure semantic similarity between response and selected context
                P, R, F1 = score([response] * len(relevant_context), relevant_context, lang='en')
                relevance_scores.append(F1.mean().item())
                
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
    def generate_evaluation_report(self,
                                 metrics: Dict[str, float],
                                 output_path: str = None) -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'analysis': {
                'factual_consistency': self._analyze_metric(
                    metrics['factual_consistency'],
                    'Factual Consistency',
                    threshold=0.7
                ),
                'coherence_loss': self._analyze_metric(
                    metrics['coherence_loss'],
                    'Coherence Loss',
                    threshold=0.3,
                    lower_is_better=True
                ),
                'context_relevance': self._analyze_metric(
                    metrics['context_relevance'],
                    'Context Relevance',
                    threshold=0.6
                )
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return self._format_report(report)
        
    def _analyze_metric(self,
                       value: float,
                       name: str,
                       threshold: float,
                       lower_is_better: bool = False) -> str:
        """Generate analysis text for a metric."""
        if lower_is_better:
            performance = "good" if value <= threshold else "needs improvement"
        else:
            performance = "good" if value >= threshold else "needs improvement"
            
        return f"{name}: {value:.3f} - Performance is {performance} " + \
               f"(threshold: {threshold})"
               
    def _format_report(self, report: Dict[str, Any]) -> str:
        """Format evaluation report as string."""
        lines = [
            "Context Retention Assistant - Evaluation Report",
            f"Generated at: {report['timestamp']}",
            "\nMetrics Summary:",
        ]
        
        for metric, value in report['metrics'].items():
            lines.append(f"- {metric}: {value:.3f}")
            
        lines.append("\nDetailed Analysis:")
        for analysis in report['analysis'].values():
            lines.append(f"- {analysis}")
            
        return "\n".join(lines) 