"""
Test suite for matrix computations in DifferentiableMakeHuman.

This module tests the five key matrix transformations:
1. matRestGlobal - rest pose computation with orthonormal bases
2. matRestRelative - relative matrices relating child to parent
3. matPoseGlobal - pose propagation for identity quaternions
4. matPoseVerts - skinning for identity pose
5. Gradient flow - backpropagation through the entire pipeline
"""
import pytest
import torch
import numpy as np
import pytest_regressions
import io
import tempfile
import PIL.Image
import sys
from pathlib import Path


class TestRestPoseComputation:
    """Test 1: Verify rest pose computation - matRestGlobal forms orthonormal bases.
    
    Validates that matRestGlobal is properly constructed after applying modifiers:
    - Orthonormality: The 3x3 rotation part forms an orthonormal basis 
      (columns are unit vectors, mutually perpendicular, determinant = 1)
    - Homogeneous structure: Bottom row is [0, 0, 0, 1] as required for 
      4x4 transformation matrices
    """
    
    def test_matRestGlobal_is_orthonormal(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that matRestGlobal contains orthonormal rotation matrices."""
        # Apply modifiers (zero) and compute rest pose
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)   
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        
        matRestGlobal = makehuman_model.matRestGlobal
        
        # Check each bone's rotation matrix
        for b in range(matRestGlobal.shape[0]):
            rotation_matrix = matRestGlobal[b, :3, :3]
            
            # Test 1: Each column should be unit length
            col_norms = torch.norm(rotation_matrix, dim=0)
            assert torch.allclose(col_norms, torch.ones(3, device=rotation_matrix.device), atol=1e-5), \
                f"Bone {b}: Column norms are not 1: {col_norms}"
            
            # Test 2: Columns should be orthogonal to each other
            # col0 · col1 = 0
            dot_01 = torch.dot(rotation_matrix[:, 0], rotation_matrix[:, 1])
            assert torch.abs(dot_01) < 1e-5, \
                f"Bone {b}: Columns 0 and 1 not orthogonal: dot={dot_01}"
            
            # col0 · col2 = 0
            dot_02 = torch.dot(rotation_matrix[:, 0], rotation_matrix[:, 2])
            assert torch.abs(dot_02) < 1e-5, \
                f"Bone {b}: Columns 0 and 2 not orthogonal: dot={dot_02}"
            
            # col1 · col2 = 0
            dot_12 = torch.dot(rotation_matrix[:, 1], rotation_matrix[:, 2])
            assert torch.abs(dot_12) < 1e-5, \
                f"Bone {b}: Columns 1 and 2 not orthogonal: dot={dot_12}"
            
            # Test 3: Matrix should be a proper rotation (det = +1)
            det = torch.det(rotation_matrix)
            assert torch.abs(det - 1.0) < 1e-5, \
                f"Bone {b}: Determinant is not 1: det={det}"
    
    def test_matRestGlobal_bottom_row(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that the bottom row of matRestGlobal is [0, 0, 0, 1]."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        
        matRestGlobal = makehuman_model.matRestGlobal
        expected_bottom_row = torch.tensor([0., 0., 0., 1.], device=matRestGlobal.device)
        
        for b in range(matRestGlobal.shape[0]):
            bottom_row = matRestGlobal[b, 3, :]
            assert torch.allclose(bottom_row, expected_bottom_row, atol=1e-6), \
                f"Bone {b}: Bottom row is not [0,0,0,1]: {bottom_row}"


class TestRelativeMatrices:
    """Test 2: Verify relative matrices - matRestRelative correctly relates child to parent.
    
    Validates that matRestRelative correctly represents parent-child bone relationships:
    - Reconstruction formula: matRestGlobal[child] = matRestGlobal[parent] × matRestRelative[child]
    - Proper format: Relative matrices are valid homogeneous transformation matrices
    """
    
    def test_matRestRelative_reconstruction(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that matRestGlobal[child] = matRestGlobal[parent] * matRestRelative[child]."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        
        matRestGlobal = makehuman_model.matRestGlobal
        matRestRelative = makehuman_model.matRestRelative
        bone_parent_indices = makehuman_model.bone_parent_indices
        
        for b in range(matRestGlobal.shape[0]):
            parent_idx = bone_parent_indices[b].item()
            
            if parent_idx == -1:
                # Root bone: matRestRelative should equal matRestGlobal
                assert torch.allclose(matRestRelative[b], matRestGlobal[b], atol=1e-5), \
                    f"Root bone {b}: matRestRelative != matRestGlobal"
            else:
                # Child bone: verify reconstruction formula
                reconstructed = torch.matmul(matRestGlobal[parent_idx], matRestRelative[b])
                assert torch.allclose(reconstructed, matRestGlobal[b], atol=1e-5), \
                    f"Bone {b}: Reconstruction failed. " \
                    f"matRestGlobal[parent] * matRestRelative != matRestGlobal[child]"
    
    def test_matRestRelative_is_homogeneous(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that matRestRelative matrices are proper homogeneous matrices."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        
        matRestRelative = makehuman_model.matRestRelative
        expected_bottom_row = torch.tensor([0., 0., 0., 1.], device=matRestRelative.device)
        
        for b in range(matRestRelative.shape[0]):
            bottom_row = matRestRelative[b, 3, :]
            assert torch.allclose(bottom_row, expected_bottom_row, atol=1e-6), \
                f"Bone {b}: matRestRelative bottom row is not [0,0,0,1]: {bottom_row}"


class TestPosePropagation:
    """Test 3: Verify pose propagation - for identity quaternions, matPoseGlobal should equal matRestGlobal.
    
    Validates the identity case - when bones aren't rotated:
    - Identity quaternions → rest pose: With no rotation, matPoseGlobal should equal matRestGlobal
    - matPose is identity: For identity quaternions, the local pose transformation should be 
      the identity matrix
    """
    
    def test_identity_pose_equals_rest_pose(self, makehuman_model, zero_modifiers, identity_quaternions):
        """For identity quaternions (no rotation), matPoseGlobal should equal matRestGlobal."""
        # Apply modifiers and compute rest pose
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        
        matRestGlobal = makehuman_model.matRestGlobal.clone()
        
        # Apply identity pose
        makehuman_model.convert_quaternions_to_pose_matrices(identity_quaternions)
        makehuman_model.compute_matPose()
        makehuman_model.compute_matPoseGlobal()
        
        matPoseGlobal = makehuman_model.matPoseGlobal
        
        # matPoseGlobal should equal matRestGlobal for identity pose
        assert torch.allclose(matPoseGlobal, matRestGlobal, atol=1e-5), \
            "Identity pose: matPoseGlobal != matRestGlobal"
    
    def test_identity_matPose_is_identity(self, makehuman_model, zero_modifiers, identity_quaternions):
        """For identity quaternions, matPose should be the identity matrix."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        makehuman_model.convert_quaternions_to_pose_matrices(identity_quaternions)
        makehuman_model.compute_matPose()
        
        matPose = makehuman_model.matPose
        identity = torch.eye(4, device=matPose.device)
        
        for b in range(matPose.shape[0]):
            assert torch.allclose(matPose[b], identity, atol=1e-5), \
                f"Bone {b}: matPose is not identity for identity quaternion"


class TestSkinning:
    """Test 4: Verify skinning - for identity pose, skinned vertices should equal modified mesh vertices.
    
    Validates Linear Blend Skinning (LBS) with identity pose:
    - No deformation: Identity pose should produce the same mesh as the modified (pre-pose) mesh
    - matPoseVerts is identity: For identity pose, the skinning transformation matrices 
      should be identity
    """
    
    def test_identity_pose_no_deformation(self, makehuman_model, zero_modifiers, identity_quaternions):
        """For identity pose, skinned mesh should equal the modified (pre-pose) mesh."""
        # Run full forward pass with identity pose
        skinned_mesh = makehuman_model.forward(zero_modifiers, identity_quaternions, with_mask=False)
        
        # The modified mesh (after modifiers, before pose) should equal skinned mesh
        # since we're applying identity pose
        modified_mesh = makehuman_model.mesh.clone()
        
        assert torch.allclose(skinned_mesh, modified_mesh, atol=1e-4), \
            "Identity pose: Skinned mesh != modified mesh"
    
    def test_matPoseVerts_identity(self, makehuman_model, zero_modifiers, identity_quaternions):
        """For identity pose, matPoseVerts should be the identity matrix."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        makehuman_model.convert_quaternions_to_pose_matrices(identity_quaternions)
        makehuman_model.compute_matPose()
        makehuman_model.compute_matPoseGlobal()
        makehuman_model.compute_matPoseVerts()
        
        matPoseVerts = makehuman_model.matPoseVerts
        identity = torch.eye(4, device=matPoseVerts.device)
        
        for b in range(matPoseVerts.shape[0]):
            assert torch.allclose(matPoseVerts[b], identity, atol=1e-4), \
                f"Bone {b}: matPoseVerts is not identity for identity pose"


class TestMatrixConsistency:
    """Additional tests to verify matrix computation consistency.
    
    Validates computational correctness and stability:
    - Deterministic: Multiple forward passes with same inputs produce identical outputs
    - Rotation validity: matPoseRotation contains proper rotation matrices 
      (orthonormal, determinant = 1)
    """
    
    def test_forward_pass_consistency(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that multiple forward passes produce consistent results."""
        # First forward pass
        output1 = makehuman_model.forward(zero_modifiers, identity_quaternions, with_mask=False)
        
        # Second forward pass with same inputs
        output2 = makehuman_model.forward(zero_modifiers, identity_quaternions, with_mask=False)
        
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Multiple forward passes produce inconsistent results"
    
    def test_matPoseRotation_is_rotation(self, makehuman_model, zero_modifiers, identity_quaternions):
        """Verify that matPoseRotation contains valid rotation matrices."""
        deps = makehuman_model.calc_modifiers_to_dependencies(zero_modifiers)
        target_weights = makehuman_model.calc_dependencies_to_targets(deps)
        makehuman_model.apply_targets(target_weights)  
        makehuman_model.update_bone_joints()
        makehuman_model.compute_rest_pose_matrices()
        makehuman_model.convert_quaternions_to_pose_matrices(identity_quaternions)
        
        matPoseRotation = makehuman_model.matPoseRotation
        
        for b in range(matPoseRotation.shape[0]):
            rotation_matrix = matPoseRotation[b, :3, :3]
            
            # Check orthonormality: R^T * R = I
            product = torch.matmul(rotation_matrix.T, rotation_matrix)
            identity = torch.eye(3, device=rotation_matrix.device)
            assert torch.allclose(product, identity, atol=1e-5), \
                f"Bone {b}: matPoseRotation is not orthonormal"
            
            # Check determinant = 1
            det = torch.det(rotation_matrix)
            assert torch.abs(det - 1.0) < 1e-5, \
                f"Bone {b}: matPoseRotation determinant is not 1: {det}"


class TestFaceMaskWithRegression:
    """Test that applying a face mask to a known body renders a consistent image using pytest-regression."""

    def test_face_mask_image_regression(self, makehuman_base_model, image_regression):
        """Snapshot test for face mask rendering."""
        import io
        from vedo import Plotter, Mesh
        import os
        import sys

        # Dynamically add the scripts directory to the system path
        scripts_path = Path(__file__).parent.parent / 'scripts' / 'functions'
        sys.path.insert(0, str(scripts_path))

        from image_visualize import vedo_show, set_camera_init_position

        # Prepare the model
        model, model_data = makehuman_base_model
        n_modifiers = model.target_delta.shape[0]
        n_bones = model.bone_parent_indices.shape[0]
        no_modifiers = model_data["example_modifiers"].to(device=model.device)
        no_quaternions = model_data["example_quaternions"].to(device=model.device)
        no_quaternions[:, 0] = 1.0  # Identity quaternions

        # Forward pass
        _ = model.forward(no_modifiers, no_quaternions, with_mask=False)

        # Save the mesh with mask applied
        with io.BytesIO() as buf:
            model.save_our_mesh(apply_mask=True, filename=buf)
            buf.seek(0)

            # Save BytesIO content to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name

            # Load the mesh into vedo
            mesh = Mesh(tmp_path)

            # Visualize using vedo
            plotter = Plotter(offscreen=True)
            set_camera_init_position(plotter, mesh, angle=30)
            plotter.add(mesh)

            # Capture the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plotter.screenshot(tmp.name)
                image = PIL.Image.open(tmp.name)

                # Convert the image to raw bytes
                with io.BytesIO() as image_bytes:
                    image.save(image_bytes, format='PNG')
                    image_bytes.seek(0)
                    image_regression.check(image_bytes.read())




class TestGradientFlow:
    """Test 5: Verify gradient flow - gradients propagate from output vertices back to inputs.
    
    Critical for neural network training - validates that gradients propagate correctly:
    - To modifiers: Gradients flow from output vertices back to modifier parameters
    - To quaternions: Gradients flow from output vertices back to bone rotations
    - Full pipeline: Both modifiers and quaternions receive valid, non-zero, finite gradients 
      simultaneously
    """
    
    def test_gradient_flow_to_modifiers(self, makehuman_base_model):
        """Verify gradients flow from output vertices back to modifiers."""
        model, model_data = makehuman_base_model

        # Create modifiers with gradients enabled
        identity_quaternions = model_data["example_quaternions"].to(device=model.device)

        modifiers = torch.randn(
            model.target_delta.shape[0], 
            device=model.device,
            requires_grad=True
        )
        
        # Forward pass
        skinned_mesh = model.forward(modifiers, identity_quaternions, with_mask=True)
        
        # Compute a simple loss (mean of all vertex positions)
        loss = skinned_mesh.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        assert modifiers.grad is not None, "No gradients for modifiers"
        assert not torch.allclose(modifiers.grad, torch.zeros_like(modifiers.grad)), \
            "Gradients for modifiers are all zero"
    
    def test_gradient_flow_to_quaternions(self, makehuman_base_model):
        """Verify gradients flow from output vertices back to quaternions."""
        model, model_data = makehuman_base_model
        defaul_modifiers = model_data["example_modifiers"].to(device=model.device)

        # Create quaternions with gradients enabled
        quaternions_raw = torch.randn(
            model.bone_parent_indices.shape[0], 4,
            device=model.device,
            requires_grad=True
        )
        # Normalize to unit quaternions
        quaternions = quaternions_raw / torch.norm(quaternions_raw, dim=1, keepdim=True)
        
        # Forward pass
        skinned_mesh = model.forward(defaul_modifiers, quaternions, with_mask=False)
        
        # Compute a simple loss
        loss = skinned_mesh.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero on the leaf tensor
        assert quaternions_raw.grad is not None, "No gradients for quaternions"
        assert not torch.allclose(quaternions_raw.grad, torch.zeros_like(quaternions_raw.grad)), \
            "Gradients for quaternions are all zero"
    
    def test_gradient_flow_full_pipeline(self, makehuman_base_model):
        """Verify gradients flow through the entire pipeline."""
        model, model_data = makehuman_base_model
        # Create inputs with gradients enabled
        modifiers_raw = torch.randn(
            model.target_delta.shape[0],
            device=model.device,
            requires_grad=True
        )
        modifiers = modifiers_raw * 0.1
        
        quaternions_raw = torch.randn(
            model.bone_parent_indices.shape[0], 4,
            device=model.device,
            requires_grad=True
        )
        quaternions = quaternions_raw / torch.norm(quaternions_raw, dim=1, keepdim=True)
        
        # Forward pass
        skinned_mesh = model.forward(modifiers, quaternions, with_mask=False)
        
        # Compute a loss (sum of squared positions)
        loss = (skinned_mesh ** 2).sum()
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist for both leaf inputs
        assert modifiers_raw.grad is not None, "No gradients for modifiers"
        assert quaternions_raw.grad is not None, "No gradients for quaternions"
        
        # Verify gradients are finite (no NaN or Inf)
        assert torch.isfinite(modifiers_raw.grad).all(), "Modifiers gradients contain NaN or Inf"
        assert torch.isfinite(quaternions_raw.grad).all(), "Quaternions gradients contain NaN or Inf"
        
        # Verify gradients are non-zero
        assert not torch.allclose(modifiers_raw.grad, torch.zeros_like(modifiers_raw.grad), atol=1e-8), \
            "Modifiers gradients are all zero"
        assert not torch.allclose(quaternions_raw.grad, torch.zeros_like(quaternions_raw.grad), atol=1e-8), \
            "Quaternions gradients are all zero"

