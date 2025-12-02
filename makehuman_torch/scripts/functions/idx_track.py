from typing import Dict, List, Union
import torch

def calc_permutation(
    local_index_map: Union[List[str], Dict[str, int]], 
    global_name_to_id: Dict[str, int]
) -> torch.Tensor:
    """
    Calculate the permutation tensor to map local indices to global indices.
    
    Args:
        local_index_map (Union[List[str], Dict[str, int]]): Local mapping of 
            category names to indices.  Can be a list of names in order or a 
            dictionary mapping names to indices.
        global_name_to_id (Dict[str, int]): Global canonical mapping of 
            category names to indices.
    
    Returns:
        torch.Tensor: A tensor of indices representing the permutation from
            local to global ordering.

    Raises:
        TypeError: If local_index_map is neither a list nor a dictionary.
        ValueError: If a category in the global map is missing from the local data.

    Example:
        >>> local_index_map = ['cat', 'elephant', 'dog', "mouse"]
        >>> global_name_to_id = {'dog': 0, 'cat': 1, 'mouse': 2, "elephant":3}
        >>> permutation = calc_permutation(
        >>>     local_index_map=local_index_map,
        >>>     global_name_to_id=global_name_to_id
        >>> )
        >>> assert permutation == torch.tensor([2, 0, 3, 1])
        >>> assert (
        >>>     np.array(local_index_map)[permutation.detach().numpy()] 
        >>>     == np.array(list(global_name_to_id.keys()))
        >>> ).all()
    """
    # 1. check if already canonical order (early exit)
    if isinstance(local_index_map, list):
        local_name_to_id = {name: i for i, name in enumerate(local_index_map)}
        
        global_names_ordered = [
            name for name, _ in sorted(global_name_to_id.items(), key=lambda item: item[1])
        ]
        
        if local_index_map == global_names_ordered:
            num_categories = len(global_names_ordered)
            return torch.arange(num_categories, dtype=torch.long)
        
    elif isinstance(local_index_map, dict):
        local_name_to_id = local_index_map
        
        if local_name_to_id == global_name_to_id:
            num_categories = len(global_name_to_id)
            return torch.arange(num_categories, dtype=torch.long)
    else:
        raise TypeError(
            "local_index_map must be a List[str] or Dict[str, int]."
        )

    # 2. if not caonical order:
    # 2.1. Initialize the permutation tensor
    num_categories = len(global_name_to_id)
    permutation_indices = torch.empty(num_categories, dtype=torch.long)
    
    # 2.2. Iterate through the canonical order to build the permutation
    for name, canonical_id in global_name_to_id.items():
        try:
            local_id = local_name_to_id[name]
            permutation_indices[canonical_id] = local_id
        except KeyError:
            raise ValueError(
                f"Category '{name}' is in the canonical map but missing from the local data."
            )

    return permutation_indices
