"""
Custom metrics for measuring coherence and factual consistency in conversations.
"""

import numpy as np
from typing import List, Dict
from bert_score import score
from rouge_score import rouge_scorer

class CoherenceMetrics:
    def __init__(self):
        """Initialize coherence metrics calculator."""
        self.rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def measure_factual_consistency(self, 
                                  base_facts: List[str],
                                  generated_responses: List[str]) -> float:
        """
        Measure factual consistency between base facts and generated responses.
        
        Args:
            base_facts: List of ground truth facts
            generated_responses: List of model-generated responses
            
        Returns:
            Consistency score between 0 and 1
        """
        # Calculate BERT score for semantic similarity
        all_scores = []
        for response in generated_responses:
            P, R, F1 = score([response] * len(base_facts), base_facts, lang='en')
            all_scores.append(F1.mean().item())
        semantic_score = np.mean(all_scores)
        
        # Calculate ROUGE scores for lexical overlap
        rouge_scores = []
        for response in generated_responses:
            for fact in base_facts:
                scores = self.rouge_calculator.score(fact, response)
                rouge_scores.append(scores['rougeL'].fmeasure)
        
        rouge_score = np.mean(rouge_scores)
        
        # Combine metrics
        consistency_score = 0.7 * semantic_score + 0.3 * rouge_score
        return consistency_score
    
    def measure_coherence_loss(self, 
                             conversation_history: List[Dict[str, str]],
                             window_size: int = 3) -> float:
        """
        Measure coherence degradation over conversation turns.
        
        Args:
            conversation_history: List of conversation turns
            window_size: Number of turns to consider for local coherence
            
        Returns:
            Coherence loss score (higher means more degradation)
        """
        if len(conversation_history) < window_size:
            return 0.0
            
        coherence_scores = []
        for i in range(len(conversation_history) - window_size + 1):
            window = conversation_history[i:i + window_size]
            local_score = self._calculate_local_coherence(window)
            coherence_scores.append(local_score)
            
        # Calculate degradation as decline in coherence scores
        degradation = np.mean(np.diff(coherence_scores))
        return max(0, degradation)  # Only consider positive degradation
        
    def _calculate_local_coherence(self, conversation_window: List[Dict[str, str]]) -> float:
        """Calculate coherence score for a local window of conversation."""
        texts = [turn['content'] for turn in conversation_window]
        
        # Calculate pairwise coherence
        scores = []
        for i in range(len(texts) - 1):
            P, R, F1 = score([texts[i]], [texts[i + 1]], lang='en')
            scores.append(F1.mean().item())
            
        return np.mean(scores) 