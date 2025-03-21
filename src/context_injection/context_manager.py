"""
Adaptive context management system for improving response coherence.
"""

from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import numpy as np
from bert_score import score
import torch
import logging

class ContextManager:
    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        """
        Initialize context manager.
        
        Args:
            model_name: Name of the model to use for tokenization
            max_length: Maximum allowed token length
            
        Raises:
            RuntimeError: If tokenizer loading fails
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_length = max_length
        except Exception as e:
            logging.error(f"Failed to load tokenizer {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to initialize ContextManager with tokenizer {model_name}")
        
    def _validate_token_length(self, text: str) -> Tuple[bool, int]:
        """
        Validate if text is within token limit.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, token_count)
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens) <= self.max_length, len(tokens)
        
    def inject_context(self,
                      current_prompt: str,
                      conversation_history: List[Dict[str, str]],
                      max_context_length: int = 512) -> str:
        """
        Inject relevant context into current prompt.
        
        Args:
            current_prompt: The current conversation prompt
            conversation_history: List of previous conversation turns
            max_context_length: Maximum number of tokens for injected context
            
        Returns:
            Enhanced prompt with injected context
            
        Raises:
            ValueError: If prompt exceeds maximum token length
        """
        # Validate prompt length
        is_valid, token_count = self._validate_token_length(current_prompt)
        if not is_valid:
            raise ValueError(
                f"Prompt exceeds maximum token length ({token_count} > {self.max_length})"
            )
            
        # Adjust max_context_length based on prompt length
        available_tokens = self.max_length - token_count
        max_context_length = min(max_context_length, available_tokens)
        
        # Select and inject context
        relevant_context = self._select_relevant_context(current_prompt, conversation_history)
        compressed_context = self._compress_context(relevant_context, max_context_length)
        enhanced_prompt = self._format_prompt_with_context(current_prompt, compressed_context)
        
        # Validate final prompt
        is_valid, token_count = self._validate_token_length(enhanced_prompt)
        if not is_valid:
            logging.warning(
                f"Enhanced prompt exceeded token limit, falling back to original prompt"
            )
            return current_prompt
            
        return enhanced_prompt
        
    def _select_relevant_context(self,
                               current_prompt: str,
                               conversation_history: List[Dict[str, str]],
                               top_k: int = 3) -> List[str]:
        """
        Select most relevant context from conversation history.
        
        Args:
            current_prompt: Current prompt to match against
            conversation_history: Previous conversation turns
            top_k: Number of most relevant turns to select
            
        Returns:
            List of selected context strings
        """
        if not conversation_history:
            return []
            
        # Calculate relevance scores using BERT Score
        history_texts = [turn['content'] for turn in conversation_history]
        _, _, F1 = score([current_prompt] * len(history_texts), history_texts, lang='en')
        relevance_scores = F1.numpy() if isinstance(F1, torch.Tensor) else F1
        
        # Select top-k most relevant turns
        top_indices = np.argsort(relevance_scores)[-top_k:]
        selected_context = [history_texts[i] for i in top_indices]
        
        return selected_context
        
    def _compress_context(self,
                         context_list: List[str],
                         max_tokens: int) -> List[str]:
        """
        Compress context to fit within token limit.
        
        Args:
            context_list: List of context strings
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Compressed list of context strings
        """
        compressed = []
        current_tokens = 0
        
        for context in context_list:
            tokens = self.tokenizer.encode(context)
            if current_tokens + len(tokens) <= max_tokens:
                compressed.append(context)
                current_tokens += len(tokens)
            else:
                # Truncate last context if needed
                available_tokens = max_tokens - current_tokens
                if available_tokens > 20:  # Only add if meaningful context can be included
                    truncated = self.tokenizer.decode(tokens[:available_tokens])
                    compressed.append(truncated)
                break
                
        return compressed
        
    def _format_prompt_with_context(self,
                                  prompt: str,
                                  context_list: List[str]) -> str:
        """
        Format prompt with injected context.
        
        Args:
            prompt: Original prompt
            context_list: List of context strings to inject
            
        Returns:
            Formatted prompt with context
        """
        if not context_list:
            return prompt
            
        context_section = "\n\nRelevant Context:\n" + "\n".join(
            f"- {context}" for context in context_list
        )
        
        return f"{context_section}\n\nCurrent Prompt:\n{prompt}" 