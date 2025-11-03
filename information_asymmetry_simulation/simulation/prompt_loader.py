"""
Prompt loader for prompt robustness testing.
Allows loading custom prompts from external files without modifying agent code.
"""

import os
from typing import Optional


class PromptLoader:
    """Loads custom prompts for agents from external files."""
    
    @staticmethod
    def load_prompt(persona: str = "neutral", custom_prompt_file: Optional[str] = None) -> Optional[str]:
        """
        Load a custom prompt from file if specified.
        
        Args:
            persona: The default persona type (neutral, competitive, uncooperative)
            custom_prompt_file: Path to custom prompt file
        
        Returns:
            Custom prompt string if file exists, None otherwise (use default)
        """
        if not custom_prompt_file:
            return None  # Use default prompt
            
        if not os.path.exists(custom_prompt_file):
            print(f"Warning: Custom prompt file '{custom_prompt_file}' not found. Using default prompt.")
            return None
            
        try:
            with open(custom_prompt_file, 'r') as f:
                prompt_content = f.read().strip()
                if prompt_content:
                    print(f"Loaded custom prompt from '{custom_prompt_file}'")
                    return prompt_content
                else:
                    print(f"Warning: Custom prompt file '{custom_prompt_file}' is empty. Using default prompt.")
                    return None
        except Exception as e:
            print(f"Error loading custom prompt from '{custom_prompt_file}': {e}. Using default prompt.")
            return None