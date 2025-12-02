import numpy as np
import re

def create_faces_to_mask_vector(
        face_groups_names, 
        face_to_group, 
        human, 
        keys = ["^joint-", "^helper-"]):
    """Create a boolean mask for faces belonging to specified groups.
    
    Args:
        face_groups_names (dict): dict mapping group names to their indices.
        face_to_group (np.ndarray): An array mapping each face to its group index.
        human: The human model object containing the mesh.
        keys (list, optional): A list of regex patterns to identify groups to 
            mask. Defaults to ["^joint-", "^helper-"].
    
    Returns:
        np.ndarray: A boolean array where True indicates the face should be 
            kept, and False indicates it should be masked.
    """
    mask = np.ones(human.mesh.getFaceCount(), dtype=bool)
    for group_name, group_idx in face_groups_names.items():
        for key in keys:
            if len(re.findall(key, group_name)) > 0:
                mask[face_to_group == group_idx] = False
    return mask