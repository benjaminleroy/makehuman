from typing import Dict, List, Tuple

def load_vocabulary(filepath: str) -> Tuple[Dict[str, int], List[str]]:
    """Load a vocabulary file mapping names to indices.

    Args:
        filepath (str): Path to the vocabulary file.

    Returns:
        Dict[str, int]: Mapping from names to indices.
        List[str]: List mapping indices back to names.
    """
    vocabulary = {}
    id_to_name = []  

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip().lower()
            if name:
                vocabulary[name] = i
                id_to_name.append(name)
    return vocabulary, id_to_name
