"""
Integration tests for the Context Retention Assistant system.
"""

import unittest
from typing import List, Dict

from src.prompt_tuning.prompt_tuner import PromptTuner
from src.metrics.coherence_metrics import CoherenceMetrics
from src.context_injection.context_manager import ContextManager
from src.evaluation.evaluator import SystemEvaluator

class TestContextRetentionSystem(unittest.TestCase):
    def setUp(self):
        """Set up test components and sample data."""
        self.prompt_tuner = PromptTuner()
        self.coherence_metrics = CoherenceMetrics()
        self.context_manager = ContextManager()
        self.evaluator = SystemEvaluator()
        
        # Sample conversation data
        self.conversation_history = [
            {
                'role': 'user',
                'content': 'What is machine learning?'
            },
            {
                'role': 'assistant',
                'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.'
            },
            {
                'role': 'user',
                'content': 'Can you explain supervised learning?'
            }
        ]
        
        self.memory_key_points = [
            'Machine learning is a subset of AI',
            'Systems learn from experience',
            'No explicit programming required'
        ]
        
        self.ground_truth_facts = [
            'Supervised learning requires labeled training data',
            'The model learns to map inputs to outputs',
            'Examples include classification and regression'
        ]
        
    def test_prompt_tuning(self):
        """Test prompt tuning functionality."""
        # Test prompt enhancement
        tuned_prompt = self.prompt_tuner.tune_prompt(
            self.conversation_history,
            self.memory_key_points
        )
        
        # Verify key components are present
        self.assertIn('Machine learning', tuned_prompt)
        self.assertIn('Key Points to Remember:', tuned_prompt)
        for point in self.memory_key_points:
            self.assertIn(point, tuned_prompt)
            
        # Test context extraction
        context = self.prompt_tuner._extract_context(self.conversation_history)
        self.assertTrue(len(context) > 0)
        self.assertIn('artificial intelligence', context)
        
    def test_coherence_metrics(self):
        """Test coherence and factual consistency metrics."""
        # Test factual consistency
        responses = [
            'Supervised learning is a type of machine learning where the model learns from labeled training data.',
            'In supervised learning, the model maps inputs to corresponding outputs based on examples.'
        ]
        
        consistency_score = self.coherence_metrics.measure_factual_consistency(
            self.ground_truth_facts,
            responses
        )
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
        
        # Test coherence loss
        coherence_loss = self.coherence_metrics.measure_coherence_loss(
            self.conversation_history
        )
        self.assertGreaterEqual(coherence_loss, 0.0)
        
    def test_context_injection(self):
        """Test context injection and management."""
        current_prompt = 'What are the differences between supervised and unsupervised learning?'
        
        # Test context injection
        enhanced_prompt = self.context_manager.inject_context(
            current_prompt,
            self.conversation_history
        )
        
        # Verify context is properly injected
        self.assertIn('Relevant Context:', enhanced_prompt)
        self.assertIn('Current Prompt:', enhanced_prompt)
        self.assertIn('machine learning', enhanced_prompt.lower())
        
        # Test context selection
        relevant_context = self.context_manager._select_relevant_context(
            current_prompt,
            self.conversation_history
        )
        self.assertTrue(len(relevant_context) > 0)
        
    def test_system_evaluation(self):
        """Test complete system evaluation."""
        test_conversations = [{
            'id': 'test1',
            'history': self.conversation_history,
            'responses': [
                'Supervised learning uses labeled data to train models.',
                'The model learns to predict outputs based on input features.'
            ]
        }]
        
        ground_truth = {
            'test1': self.ground_truth_facts
        }
        
        # Test evaluation metrics
        metrics = self.evaluator.evaluate_system(
            test_conversations,
            ground_truth
        )
        
        # Verify all expected metrics are present
        expected_metrics = ['factual_consistency', 'coherence_loss', 'context_relevance']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertGreaterEqual(metrics[metric], 0.0)
            
        # Test report generation
        report = self.evaluator.generate_evaluation_report(metrics)
        self.assertIsInstance(report, str)
        self.assertIn('Evaluation Report', report)
        self.assertIn('Metrics Summary:', report)
        self.assertIn('Detailed Analysis:', report)
        
if __name__ == '__main__':
    unittest.main() 