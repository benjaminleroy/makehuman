"""
Extract data from MakeHuman structures to initialize DifferentiableMakeHuman.

This script processes MakeHuman's internal data structures to extract:
1. Base mesh vertices
2. Modifier deltas for all modeling modifiers
3. Bone-vertex weights for LBS
4. Joint vertex indicators for bone orientation
5. Bone plane keys for orientation computation
6. Bone parent indices for hierarchy
7. Face indices for mesh topology
8. Example modifier values and bone quaternions 

The extracted data can be saved and used to initialize the DifferentiableMakeHuman
PyTorch module without requiring the full MakeHuman library at runtime.
"""

import os
import sys
import numpy as np
import torch
import sys
import pandas as pd

# Local functions -----
# Add the `functions` directory to the Python path
functions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'functions')
if functions_dir not in sys.path:
    sys.path.insert(0, functions_dir)

# Import the required modules
from torch_implimentation import DifferentiableMakeHuman
from config import main_folder_preparation
from idx_track import calc_permutation
from io_funcs import load_vocabulary
from masking import create_faces_to_mask_vector
from modifier_processing import (
    catalog_modifiers,
    create_target_dependency_dataframe,
    create_modifier_info_dataframe,
    target_to_dependencies_mapping
)

# makehuman functions -----
main_folder_preparation()

# Import MakeHuman libraries
import files3d
from human import Human
from getpath import getSysDataPath
import skeleton
from core import G
import transformations as tm
import humanmodifier
import algos3d
import wavefront


device = torch.device('cpu')

# ============================================================================
# STEP 1: Initialize MakeHuman Human object with base mesh and skeleton
# ============================================================================


# Load base mesh
base_mesh_path = os.path.abspath(getSysDataPath("3dobjs/base.obj"))
base_mesh = files3d.loadMesh(base_mesh_path)
human = Human(base_mesh)

# Initialize global G object for skeleton loading
G.app = type('obj', (object,), {'selectedHuman': human})()
G.app.progress = lambda *args, **kwargs: None
human.callEvent = lambda *args, **kwargs: None

# Load modifiers
humanmodifier.loadModifiers(getSysDataPath('modifiers/modeling_modifiers.json'), human)
humanmodifier.loadModifiers(getSysDataPath('modifiers/bodyshapes_modifiers.json'), human)
humanmodifier.loadModifiers(getSysDataPath('modifiers/measurement_modifiers.json'), human)

# Load skeleton

base_skel_path = os.path.abspath(getSysDataPath('rigs/default.mhskel'))
base_skel = skeleton.load(base_skel_path, human.meshData)
human.setBaseSkeleton(base_skel)

print("\nStep 1: Loading base mesh and skeleton...")
print(f"  Loaded mesh with {len(human.meshData.coord)} vertices")
print(f"  Loaded skeleton with {base_skel.getBoneCount()} bones")

# ============================================================================
# STEP 2: Mask collection
# ===========================================================================
print("\nStep 2: Collecting face masks for non-useful faces and vertices (e.g. hair, eyelashes, clothing)...")

face_groups_names = {str(fg.name):idx for idx, fg in enumerate(human.mesh._faceGroups)}

face_to_group = human.mesh.group

face_mask = create_faces_to_mask_vector(
    face_groups_names, 
    face_to_group, 
    human, 
    keys=["^joint-", "^helper-"]
)

# ============================================================================
# STEP 3: Extract base mesh vertices (before any modifiers)
# ============================================================================

print("\nStep 3: Extracting base mesh vertices...")

base_mesh_vertices = np.array(human.meshData.coord, dtype=np.float32)
print(f"  Base mesh shape: {base_mesh_vertices.shape}")

# Extract faces
faces = np.array(human.meshData.fvert, dtype=np.int32)
print(f"  Faces shape: {faces.shape}")



# ==========================================================================
# STEP 4: Catalog Modifiers > Dependencies > Targets + Functions
# ===========================================================================
print("\nStep 4: Cataloging modifiers...")

# Generate the catalog
modifier_catalog = catalog_modifiers(
    human=human,
    modifier_classes={
        "EthnicModifier": humanmodifier.EthnicModifier,
        "MacroModifier": humanmodifier.MacroModifier,
        "UniversalModifier": humanmodifier.UniversalModifier
        }
)

target_dependency_df = create_target_dependency_dataframe(modifier_catalog)

modifier_info_df = create_modifier_info_dataframe(modifier_catalog)

target_to_dependencies = [
    tuple(x[1]) 
    for x in target_dependency_df[["Target", "Dependency"]].drop_duplicates().iterrows()
]

modifier_to_dependencies = [
    tuple(x[1]) 
    for x in target_dependency_df[["Modifier Name", "Dependency", "Function"]].drop_duplicates().iterrows()
]

target_key_list = target_dependency_df.sort_values("Target")["Target"].unique()
dependency_key_list = target_dependency_df.sort_values("Dependency")["Dependency"].unique()
modifier_key_list = target_dependency_df.sort_values("Modifier Name")["Modifier Name"].unique()

for f,l in zip(
    ["targets.txt", "dependencies.txt", "modifiers.txt"],
    [target_key_list, dependency_key_list, modifier_key_list]
):
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'index', f), 'w') as f_out:
        for item in l:
            f_out.write(f"{item}\n")

target_key = dict(zip(target_key_list, np.arange(len(target_key_list), dtype=int)))
dependency_key = dict(zip(dependency_key_list, np.arange(len(dependency_key_list), dtype=int)))
modifier_key = dict(zip(modifier_key_list, np.arange(len(modifier_key_list), dtype=int)))

# ============================================================================
# STEP 5: Extract target to dependencies mapping (as 3D sparse COO tensor)
# ============================================================================

print("\nStep 5: Extracting target to dependencies mapping as 3D sparse tensor...")

target_to_dep_map = target_to_dependencies_mapping(
    target_key=target_key,
    dependency_key=dependency_key,
    mapping=target_to_dependencies,
    device=device
)

print(f"  Created 3D sparse target to dependencies mapping: shape {target_to_dep_map.shape}")
print(f"  Total non-zeros: {target_to_dep_map._nnz()}")
print(f"  Sparsity: {(1 - target_to_dep_map._nnz() / (np.prod(target_to_dep_map.shape))) * 100:.1f}%")


# ============================================================================
# STEP 6: Extract target modifier deltas (as 3D sparse COO tensor)
# ============================================================================

print("\nStep 6: Extracting target deltas as 3D sparse tensor...")

all_indices_list = [] # Will accumulate [target_idx, vert_idx, coord_idx] triples
all_values_list = []
target_modifiers_delta_nnz_total = 0
num_target_modifiers = len(target_key_list)
num_vertices = len(base_mesh_vertices)


for target_path in target_key_list:
    try:
        target = algos3d.getTarget(human.meshData, target_path)
        target_index = target_key[target_path]
        if target and hasattr(target, 'verts') and hasattr(target, 'data'):
            vertex_indices = target.verts  # Which vertices are affected
            translation_vectors = target.data  # Translation vectors for each affected vertex (3d)
        if vertex_indices.shape[0] > 0:
            try:
                vertex_indices = np.array(vertex_indices, dtype=np.int64)
                translation_vectors = np.array(translation_vectors, dtype=np.float32)
                
                # For each affected vertex, we need 3 entries (one per coordinate x, y, z)
                impacted_vertex_indices = np.repeat(vertex_indices, 3)  # Repeat vertex indices 3 times
                coord_indices = np.tile([0, 1, 2], vertex_indices.shape[0])  # Cycle [0,1,2] for x,y,z
                mod_indices = np.full_like(impacted_vertex_indices, target_index)  # Fill with this modifier's index
                
                # Create 3D indices: (mod_idx, vert_idx, coord_idx)
                all_indices_list.append(np.array([mod_indices, impacted_vertex_indices, coord_indices]))
                
                # Flatten translation vectors to get values
                all_values_list.append(translation_vectors.flatten())
                target_modifiers_delta_nnz_total += len(translation_vectors.flatten())

            except Exception as e:
                print(f"    Warning: Could not process {target_path}: {e}")
    except Exception as e:
            print(f"    Warning: Could not load targets for {target_path}: {e}")

# Create 3D sparse COO tensor from accumulated data
if all_indices_list:
    combined_indices = np.concatenate(all_indices_list, axis=1)  # (3, total_nnz)
    combined_values = np.concatenate(all_values_list, axis=0)  # (total_nnz,)
    
    indices_tensor = torch.tensor(combined_indices, dtype=torch.long, device=device)
    values_tensor = torch.tensor(combined_values, dtype=torch.float32, device=device)
    
    target_modifiers_delta_sparse_3d = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        (num_target_modifiers, num_vertices, 3),
        device=device
    )
else:
    # Empty sparse tensor
    target_modifiers_delta_sparse_3d = torch.sparse_coo_tensor(
        torch.zeros((3, 0), dtype=torch.long, device=device),
        torch.zeros(0, dtype=torch.float32, device=device),
        (num_target_modifiers, num_vertices, 3),
        device=device
    )

print(f"  Created 3D sparse target_modifiers deltas: shape {target_modifiers_delta_sparse_3d.shape}")
print(f"  Total non-zeros: {target_modifiers_delta_nnz_total}")
print(f"  Sparsity: {(1 - target_modifiers_delta_nnz_total / (num_target_modifiers * num_vertices * 3)) * 100:.1f}%")


# ============================================================================
# STEP 7: Extract bone-vertex weights for Linear Blend Skinning
# ============================================================================

print("\nStep 7: Extracting bone-vertex weights...")

vertex_weights_obj = base_skel.getVertexWeights()
# VertexBoneWeights.data is a dict: {bone_name: (vertex_indices_array, weights_array)}
# Convert to dense matrix format
num_vertices = len(base_mesh_vertices)
bones = base_skel.getBones()
num_bones = len(bones)

bone_vertex_weights = np.zeros((num_vertices, num_bones), dtype=np.float32)

# Get bone name to index mapping
bone_name_to_idx = {bone.name: idx for idx, bone in enumerate(bones)}

for bone_name, (vertex_indices, weights) in vertex_weights_obj.data.items():
    if bone_name in bone_name_to_idx:
        bone_idx = bone_name_to_idx[bone_name]
        # vertex_indices and weights are numpy arrays
        bone_vertex_weights[vertex_indices, bone_idx] = weights

print(f"  Bone-vertex weights shape: {bone_vertex_weights.shape}")


# ============================================================================
# STEP 8: Extract joint vertex indicators
# ============================================================================

print("\nStep 8: Extracting joint vertex indicators...")

# Joint vertex indicators map which vertices define each bone's head and tail joints
# We need to extract this from the skeleton's bone structure
bones = base_skel.getBones()
num_bones = len(bones)


joint_vertex_dict = base_skel.joint_pos_idxs
joint_indices_list = []  # Will store (vertex_idx, bone_idx, head_or_tail)

bones_id_to_names = {bone.name: idx  for idx, bone in enumerate(bones)}

for bone_idx, bone in enumerate(bones):
    # Get head and tail joint positions
    # Bones have headJoint and tailJoint that reference joint indices
    # Joints have associated vertices
    name = bone.name

    # For head joint
    if hasattr(bone, 'headJoint') and bone.headJoint is not None:
        joint_name = name + '____head'
        for v_idx in joint_vertex_dict.get(joint_name, []):
            joint_indices_list.append((v_idx, bone_idx, 0))  # 0 for head
         
    
    # For tail joint
    if hasattr(bone, 'tailJoint') and bone.tailJoint is not None:
        joint_name = name + '____tail'
        for v_idx in joint_vertex_dict.get(joint_name, []):
            joint_indices_list.append((v_idx, bone_idx, 1))  # 1 for head
# Create sparse COO tensor directly from collected indices
if len(joint_indices_list) > 0:
    # Convert list of tuples to indices tensor: (3, nnz)
    joint_indices_array = np.array(joint_indices_list, dtype=np.int64).T
    indices = torch.tensor(joint_indices_array, dtype=torch.long, device=device)
    
    # All values are 1 (binary tensor)
    values = torch.ones(len(joint_indices_list), dtype=torch.uint8, device=device)
    
    # Create sparse COO tensor
    joint_vertex_indicators = torch.sparse_coo_tensor(
        indices,
        values,
        (num_vertices, num_bones, 2),
        dtype=torch.uint8,
        device=device
    )
    
    # Calculate and print sparsity statistics
    total_elements = num_vertices * num_bones * 2
    nnz = len(joint_indices_list)
    sparsity = (1 - nnz / total_elements) * 100
    
    print(f"  Joint vertex indicators shape: {joint_vertex_indicators.shape}")
    print(f"  Non-zero entries: {nnz} / {total_elements}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Memory saved: {(total_elements - nnz) * 1 / (1024*1024):.2f} MB (compared to dense uint8)")
else:
    print("  WARNING: No non-zero entries found in joint vertex indicators!")
    joint_vertex_indicators = None

# ============================================================================
# STEP 9: Extract bone plane keys and parent indices
# ============================================================================

print("\nStep 9: Extracting bone plane keys and parent indices...")

# Bone plane key: defines 3 joints that form a plane for bone orientation
# We need to determine which joints to use for each bone's orientation
bone_plane_key = np.zeros((num_bones, 3, 2), dtype=np.int32)
bone_parent_indices = np.zeros(num_bones, dtype=np.int32)
bones_id_to_names = {bone.name: idx  for idx, bone in enumerate(bones)}


for bone_idx, bone in enumerate(bones):
    # Get parent index
    if bone.parent is not None:
        parent_idx = bones.index(bone.parent)
        bone_parent_indices[bone_idx] = parent_idx
    else:
        bone_parent_indices[bone_idx] = -1  # Root bone

    # track which joints define the plane for this bone    
    if hasattr(bone, 'roll') and hasattr(bone, 'planes'):
        plane = bone.planes[bone.roll]  
        for j_idx, j_name in enumerate(plane):
            bone_name, head_or_tail = j_name.split('____')
            bone_plane_key[bone_idx, j_idx, 0] = bones_id_to_names.get(bone_name, -999)
            bone_plane_key[bone_idx, j_idx, 1] = 0 if head_or_tail == 'head' else 1

print(f"  Bone plane key shape: {bone_plane_key.shape}")
print(f"  Bone parent indices shape: {bone_parent_indices.shape}")


# ============================================================================
# STEP 10: Prepare example modifier values and bone quaternions
# ============================================================================

print("\nStep 10: Preparing example inputs...")

# Example modifier values (matching pipeline_to_generate_a_body_mesh.py)
example_modifiers = np.zeros(len(modifier_key), dtype=np.float32)

# apply defaults first
for modifier_name, m_idx in modifier_key.items():
    value = modifier_info_df.loc[
        modifier_info_df["Modifier Name"] == modifier_name,
        "Default"
    ]
    example_modifiers[m_idx] = value.values[0]

# Set specific modifiers to match the example
if "macrodetails/Gender" in modifier_key:
    example_modifiers[modifier_key["macrodetails/Gender"]] = 1.0  # male
if "macrodetails/Age" in modifier_key:
    example_modifiers[modifier_key["macrodetails/Age"]] = 0.5  # middle age

print(f"  Example modifiers shape: {example_modifiers.shape}")
print(f"  Example modifiers non-zero proportion: {np.count_nonzero(example_modifiers) / example_modifiers.shape[0]:.3f}")

# Example bone quaternions (identity - no rotation)
example_quaternions = np.tile([1, 0, 0, 0], (num_bones, 1)).astype(np.float32)
print(f"  Example quaternions shape: {example_quaternions.shape}")


# ============================================================================
# STEP 11: Extract modifier bounds (min, max, defaults)
# ============================================================================

print("\nStep 11: Extracting modifier bounds...")

modifier_min_list = []
modifier_max_list = []
modifier_defaults_list = []

for modifier_name in modifier_key_list:
    mod_info = modifier_info_df[modifier_info_df["Modifier Name"] == modifier_name]
    
    if len(mod_info) > 0:
        # Extract min, max, and default values
        min_val = mod_info["Min"].values[0]
        max_val = mod_info["Max"].values[0]
        default_val = mod_info["Default"].values[0]
        
        modifier_min_list.append(float(min_val))
        modifier_max_list.append(float(max_val))
        modifier_defaults_list.append(float(default_val))

modifier_min = torch.tensor(modifier_min_list, dtype=torch.float32, device=device)
modifier_max = torch.tensor(modifier_max_list, dtype=torch.float32, device=device)
modifier_defaults = torch.tensor(modifier_defaults_list, dtype=torch.float32, device=device)

print(f"  Extracted bounds for {len(modifier_key_list)} modifiers:")
print(f"    Min range: [{modifier_min.min():.3f}, {modifier_min.max():.3f}]")
print(f"    Max range: [{modifier_max.min():.3f}, {modifier_max.max():.3f}]")
print(f"    Defaults range: [{modifier_defaults.min():.3f}, {modifier_defaults.max():.3f}]")


# ============================================================================
# STEP 12: Convert to PyTorch tensors and save
# ============================================================================

print("\nStep 12: Converting to PyTorch tensors and saving...")

# Convert all data to PyTorch tensors
data_dict = {
    'base_mesh': torch.tensor(base_mesh_vertices, dtype=torch.float32, device=device),
    'target_delta': target_modifiers_delta_sparse_3d,
    "modifier_to_dependency_tuples": modifier_to_dependencies,
    "dependency_to_target_map": target_to_dep_map,
    'bone_vertex_weights': torch.tensor(bone_vertex_weights, dtype=torch.float32, device=device),  # TODO: Should be sparse CSR
    'joint_vertex_indicators': joint_vertex_indicators.to_dense() if joint_vertex_indicators is not None else None,  # Convert sparse to dense for einsum
    'bone_plane_key': torch.tensor(bone_plane_key, dtype=torch.long, device=device),
    'bone_parent_indices': torch.tensor(bone_parent_indices, dtype=torch.long, device=device),
    'faces': torch.tensor(faces, dtype=torch.long, device=device),
    'face_mask': torch.tensor(face_mask, dtype=torch.bool, device=device),
    'example_modifiers': torch.tensor(example_modifiers, dtype=torch.float32, device=device),
    'modifier_min': modifier_min,
    'modifier_max': modifier_max,
    'modifier_defaults': modifier_defaults,
    'example_quaternions': torch.tensor(example_quaternions, dtype=torch.float32, device=device),
    'target_idx': target_key,
    'modifier_idx': modifier_key,
    'dependency_idx': dependency_key
}

# Save to file
output_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", '..', 'data', 'torch_model_data.pt')
)
torch.save(data_dict, output_path)
print(f"  Saved all data to: {output_path}")


# ============================================================================
# STEP 13: Test loading and initializing DifferentiableMakeHuman
# ============================================================================

print("\nStep 13: Testing model initialization...")


# Initialize model
model = DifferentiableMakeHuman(
    base_mesh=data_dict['base_mesh'],
    target_delta=data_dict['target_delta'],
    modifier_to_dependency_tuples=data_dict["modifier_to_dependency_tuples"],
    dependency_to_target_map=data_dict["dependency_to_target_map"],
    bone_vertex_weights=data_dict['bone_vertex_weights'],
    joint_vertex_indicators=data_dict['joint_vertex_indicators'],
    bone_plane_key=data_dict['bone_plane_key'],
    bone_parent_indices=data_dict['bone_parent_indices'],
    faces=data_dict['faces'],
    modifier_keys=data_dict['modifier_idx'],
    dependency_keys=data_dict['dependency_idx'],
    target_keys=data_dict['target_idx'],
    modifier_min=data_dict['modifier_min'],
    modifier_max=data_dict['modifier_max'],
    modifier_defaults=data_dict['modifier_defaults'],
    face_mask=data_dict["face_mask"],
    device=device
)

print(f"  Model initialized successfully!")
print(f"  Testing forward pass...")

# Run forward pass with example inputs
output_mesh = model.forward(
    data_dict['example_modifiers'], # TODO: this pass through won't work, and we likely need to provide the model other index structures?
    data_dict['example_quaternions'] 
)

print(f"  Forward pass successful!")
print(f"  Output mesh shape: {output_mesh.shape}")

# Save output mesh for visualization
output_mesh_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", '..', 'data', 'torch_output_mesh.obj')
)
model.save_our_mesh(apply_mask=True, filename=output_mesh_path)
print(f"  Saved output mesh to: {output_mesh_path}")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("EXTRACTION COMPLETE!")
print("="*80)
print(f"Data shapes:")
print(f"  base_mesh: {data_dict['base_mesh'].shape}")
print(f"  target_delta: {data_dict['target_delta'].shape}")
print(f"  bone_vertex_weights: {data_dict['bone_vertex_weights'].shape}")
print(f"  joint_vertex_indicators: {data_dict['joint_vertex_indicators'].shape}")
print(f"  bone_plane_key: {data_dict['bone_plane_key'].shape}")
print(f"  bone_parent_indices: {data_dict['bone_parent_indices'].shape}")
print(f"  faces: {data_dict['faces'].shape}")
print(f"  example_modifiers: {data_dict['example_modifiers'].shape}")
print(f"  example_quaternions: {data_dict['example_quaternions'].shape}")
print(f"\nSaved to: {output_path}")
print(f"Output mesh: {output_mesh_path}")

# ===========================================================================
# MakeHuman saving
# ===========================================================================

output_path_makehuman = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", '..', 'data', 'makehuman_output_mesh.obj')
)

wavefront.writeObjFile(
    output_path_makehuman,
    human.mesh
)

