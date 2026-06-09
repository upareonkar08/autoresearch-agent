import re

def count_tokens(text: str) -> int:
    """
    Rough estimation of tokens in a text string.
    Usually, 1 token is ~4 characters or ~0.75 words.
    We'll use a conservative word-based multiplier (words * 1.33).
    
    Args:
        text (str): The input text.
        
    Returns:
        int: Estimated number of tokens.
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.33)

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing duplicate spaces and newlines.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
