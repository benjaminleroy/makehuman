"""
Pytest configuration and fixtures for MakeHuman differentiable tests.
"""
import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path


@pytest.fixture
def device():
    """Get the device to run tests on (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_skeleton_data(device):
    """Create simple skeleton data for testing.
    
    Creates a minimal skeleton with 3 bones in a hierarchy:
    - Root bone (index 0, no parent)
    - Child bone 1 (index 1, parent = 0)
    - Child bone 2 (index 2, parent = 1)
    """
    num_bones = 3
    num_vertices = 100
    num_modifiers = 5
    
    # Base mesh vertices (simple grid)
    base_mesh = torch.randn(num_vertices, 3, device=device) * 0.1
    
    # Modifiers delta (sparse, mostly zeros)
    target_delta = torch.randn(num_modifiers, num_vertices, 3, device=device) * 0.01
    
    # Bone vertex weights (simple influence)
    bone_vertex_weights = torch.zeros(num_vertices, num_bones, device=device)
    bone_vertex_weights[:40, 0] = 1.0  # First 40 vertices influenced by bone 0
    bone_vertex_weights[30:70, 1] = 1.0  # Overlap region for smooth transition
    bone_vertex_weights[60:, 2] = 1.0  # Last 40 vertices influenced by bone 2
    # Normalize weights
    bone_vertex_weights = bone_vertex_weights / bone_vertex_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Joint vertex indicators (which vertices define each bone's joints)
    joint_vertex_indicators = torch.zeros(num_vertices, num_bones, 2, device=device)
    # Bone 0: head at vertices 0-9, tail at vertices 10-19
    joint_vertex_indicators[:10, 0, 0] = 1.0  # head
    joint_vertex_indicators[10:20, 0, 1] = 1.0  # tail
    # Bone 1: head at vertices 30-39, tail at vertices 40-49
    joint_vertex_indicators[30:40, 1, 0] = 1.0
    joint_vertex_indicators[40:50, 1, 1] = 1.0
    # Bone 2: head at vertices 60-69, tail at vertices 70-79
    joint_vertex_indicators[60:70, 2, 0] = 1.0
    joint_vertex_indicators[70:80, 2, 1] = 1.0
    
    # Bone plane key (define orientation plane for each bone)
    # Format: (bone_idx, head_or_tail) for 3 points
    # IMPORTANT: The three joints must be non-collinear to form a proper plane
    bone_plane_key = torch.tensor([
        [[0, 0], [0, 1], [1, 0]],  # Bone 0: head, tail, and bone 1's head (forms triangle)
        [[1, 0], [1, 1], [2, 0]],  # Bone 1: head, tail, and bone 2's head (forms triangle)
        [[2, 0], [2, 1], [1, 1]],  # Bone 2: head, tail, and bone 2's tail (forms triangle)
    ], device=device)
    
    # Parent indices (root has -1, others form a chain)
    bone_parent_indices = torch.tensor([-1, 0, 1], dtype=torch.long, device=device)
    
    # Create simple triangle faces (dummy mesh topology)
    faces = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=torch.long, device=device)
    
    # Create modifier keys (mapping modifier names to indices)
    modifier_keys = {f'modifier_{i}': i for i in range(num_modifiers)}
    
    # Create dependency keys (simple 1-to-1 mapping for testing)
    num_dependencies = num_modifiers
    dependency_keys = {f'dependency_{i}': i for i in range(num_dependencies)}
    
    # Create target keys (mapping target names to indices)
    num_targets = num_modifiers
    target_keys = {f'target_{i}': i for i in range(num_targets)}
    
    # Create modifier to dependency tuples (modifier_name, dependency_name, function_string)
    # Simple identity mapping for testing: dependency_i = modifier_i
    modifier_to_dependency_tuples = [
        (f'modifier_{i}', f'dependency_{i}', 'lambda x: x')
        for i in range(num_modifiers)
    ]
    
    # Create dependency to target mapping (sparse tensor)
    # Simple identity mapping: target_i depends on dependency_i
    rows = list(range(num_targets))
    cols = list(range(num_dependencies))
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    dependency_to_target_map = torch.sparse_coo_tensor(
        indices, values, (num_targets, num_dependencies), device=device
    )
    
    # Create modifier bounds (min, max, default)
    # Simple defaults: min=0, max=1, default=0.5 for all modifiers
    modifier_min = torch.zeros(num_modifiers, device=device)
    modifier_max = torch.ones(num_modifiers, device=device)
    modifier_defaults = torch.full((num_modifiers,), 0.5, device=device)
    
    return {
        'base_mesh': base_mesh,
        'target_delta': target_delta,
        'bone_vertex_weights': bone_vertex_weights,
        'joint_vertex_indicators': joint_vertex_indicators,
        'bone_plane_key': bone_plane_key,
        'bone_parent_indices': bone_parent_indices,
        'faces': faces,
        'modifier_keys': modifier_keys,
        'dependency_keys': dependency_keys,
        'target_keys': target_keys,
        'modifier_to_dependency_tuples': modifier_to_dependency_tuples,
        'dependency_to_target_map': dependency_to_target_map,
        'modifier_min': modifier_min,
        'modifier_max': modifier_max,
        'modifier_defaults': modifier_defaults,
        'num_bones': num_bones,
        'num_vertices': num_vertices,
        'num_modifiers': num_modifiers,
    }


@pytest.fixture
def identity_quaternions(simple_skeleton_data, device):
    """Create identity quaternions (no rotation) for all bones."""
    num_bones = simple_skeleton_data['num_bones']
    # Identity quaternion is [1, 0, 0, 0] (w, x, y, z)
    quaternions = torch.zeros(num_bones, 4, device=device)
    quaternions[:, 0] = 1.0  # w component
    return quaternions


@pytest.fixture
def zero_modifiers(simple_skeleton_data, device):
    """Create zero modifiers (no shape change)."""
    num_modifiers = simple_skeleton_data['num_modifiers']
    return torch.zeros(num_modifiers, device=device)


@pytest.fixture
def makehuman_model(simple_skeleton_data, device):
    """Create a DifferentiableMakeHuman instance with simple skeleton data."""

    
    # Add scripts/scratch to path to import the module
    scripts_path = Path(__file__).parent.parent / 'scripts' / 'functions'
    sys.path.insert(0, str(scripts_path))
    
    from torch_implimentation import DifferentiableMakeHuman
    
    model = DifferentiableMakeHuman(
        base_mesh=simple_skeleton_data['base_mesh'],
        target_delta=simple_skeleton_data['target_delta'],
        modifier_to_dependency_tuples=simple_skeleton_data['modifier_to_dependency_tuples'],
        dependency_to_target_map=simple_skeleton_data['dependency_to_target_map'],
        bone_vertex_weights=simple_skeleton_data['bone_vertex_weights'],
        joint_vertex_indicators=simple_skeleton_data['joint_vertex_indicators'],
        bone_plane_key=simple_skeleton_data['bone_plane_key'],
        bone_parent_indices=simple_skeleton_data['bone_parent_indices'],
        faces=simple_skeleton_data['faces'],
        modifier_keys=simple_skeleton_data['modifier_keys'],
        dependency_keys=simple_skeleton_data['dependency_keys'],
        target_keys=simple_skeleton_data['target_keys'],
        modifier_min=simple_skeleton_data['modifier_min'],
        modifier_max=simple_skeleton_data['modifier_max'],
        modifier_defaults=simple_skeleton_data['modifier_defaults'],
        device=device
    )
    
    return model

@pytest.fixture
def makehuman_base_model(device):
    
    # Add scripts/scratch to path to import the module
    scripts_path = Path(__file__).parent.parent / 'scripts' / 'functions'
    sys.path.insert(0, str(scripts_path))
    
    from torch_implimentation import DifferentiableMakeHuman

    # load data 
    data_path = Path(__file__).parent.parent / 'test' / 'test_data' / 'torch_model_data.pt'
    model_data = torch.load(data_path, map_location=device)

    model = DifferentiableMakeHuman(
        base_mesh=model_data["base_mesh"].to(device=device),
        target_delta=model_data["target_delta"].to(device=device),
        modifier_to_dependency_tuples=model_data.get("modifier_to_dependency_tuples", []),
        dependency_to_target_map=model_data.get("dependency_to_target_map", None),
        bone_vertex_weights=model_data["bone_vertex_weights"].to(device=device),
        joint_vertex_indicators=model_data["joint_vertex_indicators"].to(device=device),
        bone_plane_key=model_data["bone_plane_key"].to(device=device),
        bone_parent_indices=model_data["bone_parent_indices"].to(device=device),
        faces=model_data["faces"].to(device=device),
        modifier_keys=model_data.get('modifier_idx', {}),
        dependency_keys=model_data.get('dependency_idx', {}),
        target_keys=model_data.get('target_idx', {}),
        modifier_min=model_data.get("modifier_min").to(device=device) if model_data.get("modifier_min") is not None else None,
        modifier_max=model_data.get("modifier_max").to(device=device) if model_data.get("modifier_max") is not None else None,
        modifier_defaults=model_data.get("modifier_defaults").to(device=device) if model_data.get("modifier_defaults") is not None else None,
        face_mask=model_data["face_mask"].to(device=device),
        device=device)
    
    return model, model_data
    