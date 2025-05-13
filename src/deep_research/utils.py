"""
Utility functions for the deep_research package.
"""
import os
from pathlib import Path
from typing import Optional, List

def find_dotenv_files(start_dir: Optional[str] = None, max_levels_up: int = 3) -> List[str]:
    """
    Find .env files in the current directory and parent directories.
    
    Args:
        start_dir: Directory to start searching from. Defaults to current directory.
        max_levels_up: Maximum number of parent directories to search.
        
    Returns:
        List of paths to .env files, ordered from highest level (furthest parent) to lowest.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    
    dotenv_files = []
    current_dir = Path(start_dir).absolute()
    
    # Search in current directory and up to max_levels_up parent directories
    for _ in range(max_levels_up + 1):
        env_file = current_dir / '.env'
        if env_file.exists():
            dotenv_files.append(str(env_file))
        
        # Move up one directory
        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # We've reached the root directory
            break
        current_dir = parent_dir
    
    # Return files in order from highest directory to lowest
    return list(reversed(dotenv_files))

def load_dotenv_files(start_dir: Optional[str] = None, max_levels_up: int = 3) -> List[str]:
    """
    Load environment variables from .env files in current and parent directories.
    
    Variables in lower directories (closer to start_dir) take precedence over
    those in higher directories.
    
    Args:
        start_dir: Directory to start searching from. Defaults to current directory.
        max_levels_up: Maximum number of parent directories to search.
        
    Returns:
        List of paths to .env files that were successfully loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Warning: python-dotenv package is not installed. Cannot load .env files.")
        print("Install it with: pip install python-dotenv")
        return []
    
    loaded_files = []
    for dotenv_path in find_dotenv_files(start_dir, max_levels_up):
        # Load the .env file
        if load_dotenv(dotenv_path, override=True):  # override=True allows lower dirs to take precedence
            loaded_files.append(dotenv_path)
            print(f"Loaded environment variables from: {dotenv_path}")
    
    return loaded_files 