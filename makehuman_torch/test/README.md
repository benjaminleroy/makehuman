<!-- > Created by AI - not validated by human<-->
# MakeHuman Differentiable Implementation Tests

*Created by AI - not validated by a human.*

This directory contains pytest-based tests for the differentiable MakeHuman implementation.

## Test Structure

### `conftest.py`
Pytest configuration and fixtures:
- `device`: Auto-detects CUDA or falls back to CPU
- `simple_skeleton_data`: Creates a minimal 3-bone skeleton for testing
- `identity_quaternions`: Identity rotations (no pose change)
- `zero_modifiers`: Zero shape modifiers (no shape change)
- `makehuman_model`: Instantiated DifferentiableMakeHuman model

### `test_matrix_computations.py`
Tests for the five key matrix transformations:

#### 1. **TestRestPoseComputation**
- Verifies `matRestGlobal` forms orthonormal rotation matrices
- Checks column normalization, orthogonality, and determinant
- Validates bottom row is `[0, 0, 0, 1]`

#### 2. **TestRelativeMatrices**
- Verifies `matRestRelative` correctly relates child to parent
- Tests reconstruction: `matRestGlobal[child] = matRestGlobal[parent] * matRestRelative[child]`
- Validates homogeneous matrix structure

#### 3. **TestPosePropagation**
- For identity quaternions, verifies `matPoseGlobal == matRestGlobal`
- Checks that `matPose` is identity for identity quaternions
- Tests proper hierarchy propagation

#### 4. **TestSkinning**
- For identity pose, verifies skinned mesh equals modified mesh
- Checks that `matPoseVerts` is identity for identity pose
- Validates no deformation occurs with identity transformations

#### 5. **TestGradientFlow**
- Verifies gradients propagate from output vertices to modifiers
- Verifies gradients propagate from output vertices to quaternions
- Tests full pipeline gradient flow
- Validates gradients are finite and non-zero

#### 6. **TestMatrixConsistency**
- Additional consistency checks
- Tests multiple forward passes produce identical results
- Verifies `matPoseRotation` contains valid rotation matrices

## Running Tests

### Install Dependencies
```bash
# Using uv
uv pip install pytest torch pytorch3d

# Or using pip
pip install pytest torch pytorch3d
```

### Run All Tests
```bash
# From project root
pytest test/

# With verbose output
pytest test/ -v

# With coverage
pytest test/ --cov=scripts/scratch --cov-report=html
```

### Run Specific Test Classes
```bash
# Test only rest pose computation
pytest test/test_matrix_computations.py::TestRestPoseComputation -v

# Test only gradient flow
pytest test/test_matrix_computations.py::TestGradientFlow -v
```

### Run Specific Tests
```bash
# Test orthonormality
pytest test/test_matrix_computations.py::TestRestPoseComputation::test_matRestGlobal_is_orthonormal -v

# Test gradient flow to modifiers
pytest test/test_matrix_computations.py::TestGradientFlow::test_gradient_flow_to_modifiers -v
```

## Expected Test Behavior

All tests should pass if the implementation correctly:
1. Computes orthonormal rest pose matrices
2. Maintains proper parent-child relationships
3. Propagates poses through the hierarchy correctly
4. Applies skinning transformations properly
5. Maintains differentiability through backpropagation

## Troubleshooting

### Import Errors
If you get import errors for `torch_implimentation`, ensure:
- The module is in `scripts/scratch/torch_implimentation.py`
- The path is correctly added in `conftest.py`

### CUDA Errors
Tests automatically fall back to CPU if CUDA is unavailable. To force CPU:
```bash
CUDA_VISIBLE_DEVICES="" pytest test/
```

### Numerical Precision
Tests use tolerances (`atol`) appropriate for float32 precision:
- Most tests: `1e-5` to `1e-6`
- Skinning tests: `1e-4` (accumulated error through multiple operations)
- Gradient tests: `1e-8` (checking for non-zero)

## Future Test Additions

Potential additional tests:
- Test with non-trivial rotations (various quaternion values)
- Test with non-zero modifiers (shape changes)
- Test extreme cases (very small/large values)
- Performance benchmarks
- Memory usage tests
- Batch processing tests (multiple humans at once)
