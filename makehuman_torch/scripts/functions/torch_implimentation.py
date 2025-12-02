import torch 
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix
import io
from typing import List, Dict, Tuple, Optional


class DifferentiableMakeHuman(nn.Module):
    def __init__(
            self, 
            base_mesh: torch.tensor, 
            target_delta: torch.tensor, 
            modifier_to_dependency_tuples: List[Tuple[str, str, str]],
            dependency_to_target_map: torch.sparse.FloatTensor,
            bone_vertex_weights: torch.tensor, 
            joint_vertex_indicators: torch.tensor, 
            bone_plane_key: torch.tensor,
            bone_parent_indices: torch.tensor,
            faces: torch.tensor,
            modifier_keys: dict, # TODO: deal with
            dependency_keys: dict, # TODO: deal with
            target_keys: dict, # TODO: deal with
            modifier_min: Optional[torch.tensor]=None,
            modifier_max: Optional[torch.tensor]=None,
            modifier_defaults: Optional[torch.tensor]=None,
            # does bones need keys too?
            face_mask: Optional[torch.tensor]=None,
            device: Optional[torch.device]=torch.device('cpu')
        ):
        """Create the foundations of a MakeHuman object, including 
        information on the body mesh and initial skeleton structure.

        Args:
            base_mesh (torch.tensor): initial body mesh vertex positions in
                the rest pose. (v,3)
            target_delta (torch.tensor): sparse displaement vector for body 
                mesh point v due to modifier m. Assumed to be bound betwen 0 
                and 1 (inclusive). (m,v,3)
            bone_vertex_weights (torch.tensor): sparse matrix mapping the 
                influencing of bones movement onto mesh point vertices. 
                Assumed to be bound between 0 and 1 (inclusive) and per v, the 
                values sum to 1. (v,b)
            joint_vertex_indicators (torch.tensor): sparse binary matrix mapping
                the influencing of vertices position to the likely joint points
                via a non-weighted average. (v, b, 2). The last dimension is 
                for head and tail joint positions of each bone, respectively.
            bone_plane_key (torch.tensor): Mapping of bones to those joints that
                together make a plan for the orientation of the bone. (b, 3, 2)
                The 3 points are identified with the bone number and the head
                or tail joint (0 for head, 1 for tail).
            bone_parent_indices (torch.array): Parent indices for each bone. (b,)
                If a bone has no parent (aka the root bone), the index is set to -1.
            faces (torch.tensor or np.ndarray): Face indices (triangles). (f, 3)
                Required for saving meshes and computing vertex normals. Not
                required for deformation / forward pass.
            modifier_min (torch.tensor, optional): Minimum values for each 
                modifier. (m,) If not provided, defaults to zeros.
            modifier_max (torch.tensor, optional): Maximum values for each 
                modifier. (m,) If not provided, defaults to ones.
            modifier_defaults (torch.tensor, optional): Default/neutral values 
                for each modifier. (m,) If not provided, defaults to 
                (min + max) / 2.
            face_mask (torch.tensor, optional): Binary mask indicating which
                faces to include in the final output mesh. (f,)
            device (torch.device, optional): Device to store tensors on. Defaults to CPU.
        """
        super(DifferentiableMakeHuman, self).__init__()
        self.device = device
        self.base_mesh = base_mesh  # (19158, 3)
        self.target_delta = target_delta  # (249, 19158, 3)
        self.bone_vertex_weights = bone_vertex_weights  # (19158, 163)
        self.joint_vertex_indicators = joint_vertex_indicators  # (19158, 163, 2)
        self.bone_plane_key = bone_plane_key  # (163, 3, 2)
        self.modifier_keys = modifier_keys
        self.dependency_keys = dependency_keys
        self.target_keys = target_keys
        self.modifier_to_dependency_tuples = modifier_to_dependency_tuples
        self.dependency_to_target_map = dependency_to_target_map

        # Register modifier bounds as buffers (non-trainable tensors)
        num_modifiers = target_delta.shape[0]
        if modifier_min is not None:
            self.register_buffer(
                'modifier_min', modifier_min.to(device=device)
            )
        else:
            self.register_buffer(
                'modifier_min', torch.zeros(num_modifiers, device=device)
            )
            
        if modifier_max is not None:
            self.register_buffer(
                'modifier_max', modifier_max.to(device=device)
            )
        else:
            self.register_buffer(
                'modifier_max', torch.ones(num_modifiers, device=device)
            )
            
        if modifier_defaults is not None:
            self.register_buffer(
                'modifier_defaults', modifier_defaults.to(device=device)
            )
        else:
            self.register_buffer(
                'modifier_defaults', (self.modifier_min + self.modifier_max) / 2
            )

        # bone ordering check
        if not torch.all(
            bone_parent_indices 
            < torch.arange(
                bone_parent_indices.shape[0], 
                device=self.device)
            ):
            raise ValueError(
                "bone_parent_indices must be sorted such that each bone's "
                + "parent index is less than the bone's index (we expect "
                + "breadth first structure)."
        )
        self.bone_parent_indices = bone_parent_indices  # (163,)
        # Store faces as a buffer (won't be trained, but will move with model to device)
        if isinstance(faces, torch.Tensor):
            self.register_buffer('faces', faces.long())
        else:
            self.register_buffer('faces', torch.tensor(faces, dtype=torch.long))
        if face_mask is not None:
            self.face_mask = face_mask
        else:
            self.face_mask = torch.ones(self.faces.shape[0], dtype=torch.bool, device=self.device)



        # Modified mesh state
        self.mesh = None  # (19158, 3) - mesh after applying modifiers
        self.joints = None  # (163, 2, 3) - joint positions after modifiers
        
        # Rest pose matrices (after modifiers, before pose)
        self.matRestGlobal = None  # (163, 4, 4) - rest pose in global coords
        self.matRestRelative = None  # (163, 4, 4) - rest pose relative to parent
        
        # Pose matrices (during pose application)
        # Note: matPoseRotation is the raw rotation in world space from quaternions
        #       matPose is the local transformation in bone's rest space (used for hierarchy)
        self.matPoseRotation = None  # (163, 4, 4) - rotation from quaternions (world space)
        self.matPose = None  # (163, 4, 4) - local pose transformation (rest space)
        self.matPoseGlobal = None  # (163, 4, 4) - global pose transformation
        self.matPoseVerts = None  # (163, 4, 4) - skinning transformation

        self.processed_mesh = None  # (v, 3) - final deformed mesh after forward pass

    def clamp_modifiers(self, modifiers: torch.tensor) -> torch.tensor:
        """Clamp modifier values to their valid [min, max] ranges.
        
        Args:
            modifiers (torch.tensor): Modifier values to clamp. (m,)
            
        Returns:
            torch.tensor: Clamped modifier values within [modifier_min, 
            modifier_max]. (m,)
        """
        return torch.clamp(
            modifiers, 
            min=self.modifier_min, 
            max=self.modifier_max
        )

    def forward(self, modifiers, bone_quaterions, with_mask=True):
        """forward apply the modifiers and bone transformations to deform
        the body mesh.

        Args:
            modifiers (torch.array): scalar values for each modifier m, 
                representing its influence on the body mesh. (m,)
            bone_quaterions (torch.tensor): normalized quaternion rotations 
                for each bone b, representing its rotation from the rest pose. 
                (b, 4)

        Returns:
            torch.tensor: deformed body mesh vertex positions after applying the
                modifiers and bone transformations. (v, 3)
        """
        self.processed_mesh = None

        # Clamp modifiers to valid [min, max] range
        modifiers = self.clamp_modifiers(modifiers)

        # Step 1: Apply modifiers and update rest pose structure
        deps = self.calc_modifiers_to_dependencies(modifiers)
        target_weights = self.calc_dependencies_to_targets(deps)
        self.apply_targets(target_weights)
        self.update_bone_joints()
        self.compute_rest_pose_matrices()  # Computes matRestGlobal and matRestRelative

        # Step 2: Convert quaternions to pose matrices and compute posed skeleton
        self.convert_quaternions_to_pose_matrices(bone_quaterions)
        self.compute_matPose()  # Compute local pose transformation
        self.compute_matPoseGlobal()  # Propagate through hierarchy
        self.compute_matPoseVerts()  # Compute skinning matrices

        # Step 3: Apply Linear Blend Skinning to deform the mesh
        self.processed_mesh = self.apply_linear_blend_skinning()

        if with_mask:
            info = self.apply_face_mask(self.processed_mesh, self.face_mask)
            return info["vertices"]
        return self.processed_mesh


    def calc_modifiers_to_dependencies(self,
            modifiers: torch.tensor
        ):
        """Convert of modifier values to a vector of dependencies.
        
        Args:
            modifiers (torch.tensor): scalar values for each modifier m, 
                representing its influence on the body mesh. (m,) Should be full
                of values (including default values).

        Returns:
            torch.tensor: scalar values for each dependency d as a function of 
                the modifier values. In order of the dependency_key
        """
        dependency_key = self.dependency_keys
        modifier_key = self.modifier_keys
        mapping = self.modifier_to_dependency_tuples

        dependencies = torch.ones(len(dependency_key), device=self.device)
        for mod, dep, function in mapping:
            if mod not in modifier_key:
                raise ValueError(
                    f"Modifier {mod} not found in modifier_key."
                )
            if dep not in dependency_key:
                raise ValueError(
                    f"Dependency {dep} not found in dependency_key."
                )

            i_d, i_m = dependency_key[dep], modifier_key[mod]

            dependencies[i_d] = eval(function)(modifiers[i_m])

        return dependencies

    def calc_dependencies_to_targets(
            self,
            dependencies: torch.tensor, 
            ):
        """Convert of dependency values to a vector of targets.
        
        Args:
            dependencies (torch.tensor): scalar values for each dependency d. (d,)

        Returns:
            torch.tensor: scalar values for each target t as a function of 
                the dependency values. In order of the target_key (t,)
        """
        mapping = self.dependency_to_target_map
        log_dependencies = torch.log(dependencies).unsqueeze(1)
        targets_log = torch.sparse.mm(mapping, log_dependencies) # assumed to handle -inf well...
        targets = torch.exp(targets_log).squeeze(1)

        return targets

    def apply_targets(self, target_weights):
        """Apply the given target_weights to the base mesh to get the modified mesh.

        Args:
            target_weights (torch.array): scalar values for each modifier t, 
                representing its influence on the body mesh. (t,)

        Updates:
            self.mesh: modified body mesh vertex positions after applying 
                the modifiers. (v, 3)
        """
        # Apply modifiers to base mesh
        if self.target_delta.is_sparse:
            # For sparse tensor: weight by targets and keep sparse
            coo = self.target_delta.coalesce()
            indices = coo.indices()  # (3, nnz) - [mod_idx, vert_idx, coord_idx]
            values = coo.values()    # (nnz,)
            
            # Weight each value by its targets's intensity
            target_indices = indices[0]
            weighted_values = values * target_weights[target_indices]
            
            # Sum over target dimension (dimension 0) to get (num_vertices, 3)
            # We need to create a new sparse tensor with indices for just (vert_idx, coord_idx)
            new_indices = indices[1:]  # Drop mod_idx dimension
            delta_sum_sparse = torch.sparse_coo_tensor(
                new_indices, weighted_values, 
                (self.target_delta.shape[1], self.target_delta.shape[2]),
                device=self.device
            )
            
            # Convert to dense for mesh addition (since base_mesh is dense)
            delta_sum = delta_sum_sparse.to_dense()
        else:
            # Dense path using einsum
            delta_sum = torch.einsum('m,mvc->vc', target_weights, self.target_delta)
        
        self.mesh = self.base_mesh + delta_sum

    def update_bone_joints(self):
        """Update the bone joint positions based on the modified mesh.

        Updates:
            self.joints: updated joint positions after applying modifiers.
        """
        # Update joint positions as unweighted average of associated vertices
        # Head joints: (b, 3)
        head_sum = torch.einsum('vb,v,vc->bc', self.joint_vertex_indicators[:, :, 0], 
                    torch.ones(self.mesh.shape[0], device=self.device), self.mesh)
        head_count = torch.sum(self.joint_vertex_indicators[:, :, 0], dim=0, keepdim=True).T
        joint_positions_head = head_sum / head_count.clamp(min=1e-8)
        
        # Tail joints: (b, 3)
        tail_sum = torch.einsum('vb,v,vc->bc', self.joint_vertex_indicators[:, :, 1], 
                    torch.ones(self.mesh.shape[0], device=self.device), self.mesh)
        tail_count = torch.sum(self.joint_vertex_indicators[:, :, 1], dim=0, keepdim=True).T
        joint_positions_tail = tail_sum / tail_count.clamp(min=1e-8)
        
        self.joints = torch.stack([joint_positions_head, joint_positions_tail], dim=1)

    def compute_rest_pose_matrices(self):
        """Compute matRestGlobal and matRestRelative after modifiers are applied.

        Updates:
            self.matRestGlobal: Global rest pose after modifiers. (b, 4, 4)
            self.matRestRelative: Rest pose relative to parent. (b, 4, 4)
        """
        # Compute matRestGlobal (global rest pose after modifiers) --------
        # Build list of matrices to avoid in-place operations (for autograd)
        rest_matrices = []
        
        for b in range(len(self.bone_plane_key)):
            j1, j2, j3 = self.bone_plane_key[b]
            # Compute bone direction (head to tail)
            bone_direction = self.joints[j1[0], j1[1], :] - self.joints[j2[0], j2[1], :]

            bone_direction = bone_direction / torch.norm(bone_direction)
            
            # Compute normal vector from the plane defined by three joints
            # j1, j2, j3 are (bone_idx, head_or_tail) pairs
            v1 = self.joints[j2[0], j2[1], :] - self.joints[j1[0], j1[1], :]
            v2 = self.joints[j3[0], j3[1], :] - self.joints[j2[0], j2[1], :]
            normal = torch.linalg.cross(v2/torch.norm(v2), v1/torch.norm(v1))
            normal = normal / torch.norm(normal)
            
            # Compute the third axis (perpendicular to normal and bone_direction)
            third_axis = torch.linalg.cross(normal, bone_direction)
            third_axis = third_axis / torch.norm(third_axis)
            
            # computer orthonormal basis correction to normal
            normal_star = torch.linalg.cross(bone_direction, third_axis)
            normal_star = normal_star / torch.norm(normal_star)

            # Construct the rest pose matrix (orthonormal basis + translation)
            # Build matrix using torch.cat to avoid in-place operations
            rotation = torch.stack([normal_star, bone_direction, third_axis], dim=1)  # (3, 3)
            translation = self.joints[j1[0], j1[1], :].unsqueeze(1)  # (3, 1) - head joint position
            
            # Build 4x4 homogeneous matrix
            top_row = torch.cat([rotation, translation], dim=1)  # (3, 4)
            bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=rotation.dtype).to(self.device)  # (1, 4)
            mat = torch.cat([top_row, bottom_row], dim=0)  # (4, 4)
            rest_matrices.append(mat)
        
        matRestGlobal = torch.stack(rest_matrices, dim=0)  # (b, 4, 4)
        self.matRestGlobal = matRestGlobal

        # Compute matRestRelative (rest pose relative to parent) ---------
        relative_matrices = []
        for b in range(len(self.bone_parent_indices)):
            parent_idx = self.bone_parent_indices[b]
            if parent_idx == -1:  # Root bone
                relative_matrices.append(matRestGlobal[b])
            else:
                assert torch.det(matRestGlobal[parent_idx]) > 1e-8, (
                    f"Singular parent rest matrix for bone {b} with parent {parent_idx}"
                )

                mat_rel = torch.matmul(
                    torch.linalg.inv(matRestGlobal[parent_idx]),
                    matRestGlobal[b]
                )
                relative_matrices.append(mat_rel)
        
        self.matRestRelative = torch.stack(relative_matrices, dim=0)  # (b, 4, 4)

    def convert_quaternions_to_pose_matrices(self, bone_quaternions):
        """Convert bone quaternions to 4x4 pose rotation matrices (matPoseRotation).
        
        This creates the raw rotation matrices in world/absolute space from the input
        quaternions. These represent the DESIRED rotations, not yet relative to the
        bone's rest pose.
        
        Args:
            bone_quaternions (torch.tensor): quaternion rotations for each bone (b, 4)
        
        Updates:
            self.matPoseRotation: 4x4 rotation matrices in world space (b, 4, 4)
        """
        # Convert quaternions to 3x3 rotation matrices
        rotation_3x3 = quaternion_to_matrix(bone_quaternions)  # (b, 3, 3)
        
        # Build 4x4 homogeneous matrices without in-place operations
        num_bones = bone_quaternions.shape[0]
        
        # Create matrices by concatenating instead of in-place assignment
        matrices = []
        for i in range(num_bones):
            # Build each 4x4 matrix individually
            top_row = torch.cat([rotation_3x3[i], torch.zeros(3, 1, device=self.device, dtype=rotation_3x3.dtype)], dim=1)  # (3, 4)
            bottom_row = torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=rotation_3x3.dtype)  # (1, 4)
            mat = torch.cat([top_row, bottom_row], dim=0)  # (4, 4)
            matrices.append(mat)
        
        self.matPoseRotation = torch.stack(matrices, dim=0)  # (b, 4, 4)

    def compute_matPose(self):
        """Compute matPose (local pose transformation relative to rest pose).
        
        This transforms matPoseRotation (world space rotation) into the bone's local
        rest space. This is crucial because:
        - matPoseRotation says: "I want the bone rotated this way in world space"
        - matPose says: "Here's how to change the bone from its rest orientation"
        
        The local representation (matPose) is what allows proper hierarchical 
        propagation through parent-child bone relationships.
                
        Updates:
            self.matPose: Local pose transformation in bone's rest space (b, 4, 4)
        """
        inv_rest = torch.linalg.inv(self.matRestGlobal)
        self.matPose = torch.matmul(
            torch.matmul(inv_rest, self.matPoseRotation),
            self.matRestGlobal
        )

    def compute_matPoseGlobal(self):
        """Compute matPoseGlobal (global pose transformation).
        
        This propagates the pose through the bone hierarchy by combining:
        - Parent's global pose (accumulated from root)
        - Bone's rest pose relative to parent (matRestRelative)
        - Bone's local pose transformation (matPose, not matPoseRotation!)
        
        Note: We use matPose here (the local transformation), not matPoseRotation
        (the world space rotation). This is essential for proper hierarchy propagation.
                
        Updates:
            self.matPoseGlobal: Global pose for each bone (b, 4, 4)
        """
        # Build list to avoid in-place operations (for autograd)
        pose_global_matrices = []
        
        for b in range(len(self.bone_parent_indices)):
            parent_idx = self.bone_parent_indices[b]
            if parent_idx == -1:  # Root bone
                mat = torch.matmul(
                    self.matRestRelative[b],
                    self.matPose[b]  # Using matPose, not matPoseRotation
                )
            else:
                mat = torch.matmul(
                    pose_global_matrices[parent_idx],
                    torch.matmul(
                        self.matRestRelative[b], 
                        self.matPose[b])
                )
            pose_global_matrices.append(mat)
        
        self.matPoseGlobal = torch.stack(pose_global_matrices, dim=0)

    def compute_matPoseVerts(self):
        """Compute matPoseVerts (skinning transformation matrix).
                
        This matrix transforms vertices from rest pose to posed position.
        """
        inv_rest = torch.linalg.inv(self.matRestGlobal)
        self.matPoseVerts = torch.matmul(self.matPoseGlobal, inv_rest)

    def apply_linear_blend_skinning(self):
        """Apply Linear Blend Skinning (LBS) to deform the mesh based on the
        skinning transformation matrices (matPoseVerts) and bone vertex weights.

        Returns:
            torch.tensor: deformed body mesh vertex positions after applying LBS. (v, 3)
        """
        # Convert mesh to homogeneous coordinates
        v_homogeneous = torch.cat(
            [self.mesh, torch.ones((self.mesh.shape[0], 1), device=self.device)], dim=1
        )  # (v, 4)

        deformed_vertices = torch.zeros_like(self.mesh)

        for b in range(len(self.bone_parent_indices)):
            weight = self.bone_vertex_weights[:, b].unsqueeze(1)  # (v, 1)
            # Apply skinning transformation: matPoseVerts * vertex
            transformed_v = torch.matmul(
                self.matPoseVerts[b, :, :], 
                v_homogeneous.T
            ).T  # (v, 4)
            deformed_vertices += weight * transformed_v[:, :3]  # Weighted sum

        return deformed_vertices
    
    def apply_face_mask(self, mesh, face_mask):
        """Filter mesh to only include visible faces and their associated vertices.
        
        This replicates module3d.Object3D.filterMaskedVerts() to handle hidden faces
        (e.g., clothing hidden under body mesh).
        
        Args:
            face_mask (torch.Tensor): Boolean mask indicating which faces are visible. (f,)
            
        Returns:
            tuple: (filtered_vertices, filtered_faces, vertex_mapping)
                - filtered_vertices: Only vertices used by visible faces (v_filtered, 3)
                - filtered_faces: Face indices remapped to filtered vertices (f_filtered, 3)
                - vertex_parent_map: Maps filtered vertex idx to original vertex idx (v_filtered,)
        """
        # Get all unique vertices used by visible faces
        visible_faces = self.faces[face_mask]  # (f_visible, 3)
        unique_verts = torch.unique(visible_faces.flatten()).sort()[0]  # (v_filtered,)
        
        # Create mapping: original_vert_idx -> filtered_vert_idx
        vertex_map = torch.full((mesh.shape[0],), -1, dtype=torch.long, device=self.device)
        vertex_map[unique_verts] = torch.arange(len(unique_verts), device=self.device)
        
        # Filter vertices
        filtered_vertices = mesh[unique_verts]  # (v_filtered, 3)
        
        # Remap face indices to filtered vertex indices
        remapped_faces = vertex_map[visible_faces]  # (f_visible, 3)
        
        # Remap bone weights to filtered vertices
        filtered_bone_weights = self.bone_vertex_weights[unique_verts, :]  # (v_filtered, b)
        
        # Remap joint indicators to filtered vertices
        filtered_joint_indicators = self.joint_vertex_indicators[unique_verts, :, :]  # (v_filtered, b, 2)
        
        # Remap modifier deltas to filtered vertices
        if self.target_delta.is_sparse:
            # For sparse tensors, filter by vertex index
            coo = self.target_delta.coalesce()
            indices = coo.indices()  # (3, nnz)
            values = coo.values()
            
            # Keep only entries for visible vertices
            vert_indices = indices[1]
            valid_mask = vertex_map[vert_indices] >= 0
            
            filtered_indices = indices[:, valid_mask]
            filtered_values = values[valid_mask]
            
            # Remap vertex indices
            filtered_indices[1] = vertex_map[filtered_indices[1]]
            
            filtered_target_delta = torch.sparse_coo_tensor(
                filtered_indices,
                filtered_values,
                (self.target_delta.shape[0], len(unique_verts), self.target_delta.shape[2]),
                device=self.device
            )
        else:
            filtered_target_delta = self.target_delta[:, unique_verts, :]
        
        return {
            'vertices': filtered_vertices,
            'faces': remapped_faces,
            'bone_vertex_weights': filtered_bone_weights,
            'joint_vertex_indicators': filtered_joint_indicators,
            'target_delta': filtered_target_delta,
            'vertex_parent_map': unique_verts,  # Maps filtered idx -> original idx
            'vertex_map': vertex_map,  # Maps original idx -> filtered idx (-1 if not visible)
        }



    @staticmethod
    def _compute_vertex_normals(vertices, faces):
        """Compute per-vertex normals from mesh geometry.
        
        This replicates the calcVertexNormals() method from module3d.Object3D.
        The algorithm:
        1. Compute face normals (unnormalized for area-weighting)
        2. For each vertex, accumulate face normals from all adjacent faces
        3. Normalize the accumulated vertex normals
        
        Args:
            vertices (torch.Tensor): Vertex positions. (v, 3)
            faces (torch.Tensor): Face indices. (f, 3) or (f, 4)
            
        Returns:
            torch.Tensor: Vertex normals. (v, 3)
        """
        verts_per_face = faces.shape[1]
        
        # Step 1: Compute face normals (unnormalized for area-weighting)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.linalg.cross(edge1, edge2, dim=1)
        
        # Step 2: Accumulate face normals to vertices
        normals = torch.zeros_like(vertices)
        for i in range(verts_per_face):
            normals.index_add_(0, faces[:, i], face_normals)
        
        # Step 3: Normalize vertex normals
        normals = normals / torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
        
        return normals
    
    def save_our_mesh(self, apply_mask=True, filename=""):
        """Save the given vertices as an OBJ file for visualization.
        
        This matches the output of wavefront.writeObjFile used in the MakeHuman pipeline.

        Args:
            vertices (torch.Tensor): Vertex positions to save. (v, 3)
            filename (str): Path to save the OBJ file.
        """
        if self.processed_mesh is None:
            raise ValueError("Processed mesh is None. Run forward() before saving the mesh.")

        if not apply_mask:
            vertices = self.processed_mesh
            faces = self.faces
        else:
            # Apply face mask
            info = self.apply_face_mask(self.processed_mesh, self.face_mask)
            vertices = info["vertices"]
            faces = info["faces"]

        vertices_np = vertices.detach().cpu().numpy()
        faces_np = faces.cpu().numpy()

        
        # Compute vertex normals (required for proper rendering)
        vertex_normals = self._compute_vertex_normals(vertices, faces)
        vertex_normals_np = vertex_normals.detach().cpu().numpy()
        
        def write_obj(f):
            def write_line(line):
                if isinstance(f, io.BytesIO):
                    f.write(line.encode())
                else:
                    f.write(line)

            # Write header
            write_line("# OBJ file generated by torch_implementation.py\n")
            write_line(f"# Vertices: {len(vertices_np):,}\n")
            write_line(f"# Faces: {len(faces_np):,}\n\n")

            # Write vertices
            for v in vertices_np:
                write_line(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write vertex normals
            for n in vertex_normals_np:
                write_line(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces with vertex//normal indices (OBJ uses 1-based indexing)
            for face in faces_np:
                write_line(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1} {face[3]+1}//{face[3]+1}\n")

        if isinstance(filename, io.BytesIO):
            write_obj(filename)
        else:
            with open(filename, 'w') as f:
                write_obj(f)






# def save_our_mesh(vertices, faces, filename):
#     """Save the given vertices as an OBJ file for visualization.
    
#     This matches the output of wavefront.writeObjFile used in the MakeHuman pipeline.

#     Args:
#         vertices (torch.Tensor): Vertex positions to save. (v, 3)
#         filename (str): Path to save the OBJ file.
#     """
#     vertices_np = vertices.detach().cpu().numpy()
#     faces_np = faces.cpu().numpy()
    
#     # Compute vertex normals (required for proper rendering)
#     vertex_normals = _compute_vertex_normals(vertices, faces)
#     vertex_normals_np = vertex_normals.detach().cpu().numpy()
    
#     with open(filename, 'w') as f:
#         # Write header
#         f.write("# OBJ file generated by torch_implementation.py\n")
#         f.write(f"# Vertices: {len(vertices_np):,}\n")
#         f.write(f"# Faces: {len(faces_np):,}\n\n")
        
#         # Write vertices
#         for v in vertices_np:
#             f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
#         # Write vertex normals
#         for n in vertex_normals_np:
#             f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
#         # Write faces with vertex//normal indices (OBJ uses 1-based indexing)
#         for face in faces_np:
#             f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1} {face[3]+1}//{face[3]+1}\n")




# def _compute_vertex_normals(vertices, faces):
#     """Compute per-vertex normals from mesh geometry.
    
#     This replicates the calcVertexNormals() method from module3d.Object3D.
#     The algorithm:
#     1. Compute face normals (unnormalized for area-weighting)
#     2. For each vertex, accumulate face normals from all adjacent faces
#     3. Normalize the accumulated vertex normals
    
#     Args:
#         vertices (torch.Tensor): Vertex positions. (v, 3)
#         faces (torch.Tensor): Face indices. (f, 3) or (f, 4)
        
#     Returns:
#         torch.Tensor: Vertex normals. (v, 3)
#     """
#     verts_per_face = faces.shape[1]
    
#     # Step 1: Compute face normals (unnormalized for area-weighting)
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]
    
#     edge1 = v1 - v0
#     edge2 = v2 - v0
#     face_normals = torch.linalg.cross(edge1, edge2, dim=1)
    
#     # Step 2: Accumulate face normals to vertices
#     normals = torch.zeros_like(vertices)
#     for i in range(verts_per_face):
#         normals.index_add_(0, faces[:, i], face_normals)
    
#     # Step 3: Normalize vertex normals
#     normals = normals / torch.norm(normals, dim=1, keepdim=True).clamp(min=1e-8)
    
#     return normals


# def load_vocabulary(filepath: str) -> Tuple[Dict[str, int], List[str]]:
#     """Load a vocabulary file mapping names to indices.

#     Args:
#         filepath (str): Path to the vocabulary file.

#     Returns:
#         Dict[str, int]: Mapping from names to indices.
#         List[str]: List mapping indices back to names.
#     """
#     vocabulary = {}
#     id_to_name = []  

#     with open(filepath, 'r') as f:
#         for i, line in enumerate(f):
#             name = line.strip().lower()
#             if name:
#                 vocabulary[name] = i
#                 id_to_name.append(name)
#     return vocabulary, id_to_name

#^ this in a config file?
# set defaults for each thing (e.g. TARGET_MAPPING, DEPENDENCY_MAPPING, MODIFIER_MAPPING)
# ^ then do a calc_permutation(local, global) function to get the correct ordering for each run
# do it internally for each function? 
# # probably could do a bas
# def calc_permutation(
#     local_index_map: Union[List[str], Dict[str, int]], 
#     global_name_to_id: Dict[str, int]
# ) -> torch.Tensor:
#     # 1. Standardize the local map into a dictionary: name -> local_id
#     if isinstance(local_index_map, list):
#         # Create a quick lookup for the local order for the main loop
#         local_name_to_id = {name: i for i, name in enumerate(local_index_map)}
        
#         # --- EARLY OPTIMIZATION/VALIDATION CHECK ---
#         # If the local list is identical to the global order (i.e., list of names 
#         # extracted from the global dict by index), no permutation is needed.
#         # This requires reconstructing the global list for comparison:
#         global_names_ordered = [
#             name for name, _ in sorted(global_name_to_id.items(), key=lambda item: item[1])
#         ]
        
#         if local_index_map == global_names_ordered:
#             # Return the identity permutation: [0, 1, 2, ...]
#             num_categories = len(global_names_ordered)
#             return torch.arange(num_categories, dtype=torch.long)
        
#     elif isinstance(local_index_map, dict):
#         local_name_to_id = local_index_map
        
#         # --- EARLY VALIDATION CHECK (Dictionaries) ---
#         # If the keys and values are identical between the local and global dictionaries,
#         # it means the mapping is already correct. This is the simplest check.
#         if local_name_to_id == global_name_to_id:
#             # Return the identity permutation
#             num_categories = len(global_name_to_id)
#             return torch.arange(num_categories, dtype=torch.long)
#     else:
#         raise TypeError(
#             "local_index_map must be a List[str] or Dict[str, int]."
#         )

#     # If the check fails, proceed with the full calculation (Steps 2 & 3)
#     # ----------------------------------------------------------------------

#     # 2. Initialize the permutation tensor
#     num_categories = len(global_name_to_id)
#     permutation_indices = torch.empty(num_categories, dtype=torch.long)
    
#     # 3. Iterate through the CANONICAL order to build the permutation
#     for name, canonical_id in global_name_to_id.items():
#         try:
#             local_id = local_name_to_id[name]
#             permutation_indices[canonical_id] = local_id
#         except KeyError:
#             raise ValueError(
#                 f"Category '{name}' is in the canonical map but missing from the local data."
#             )

#     return permutation_indices



# def target_to_dependencies_mapping(
#         target_key: Dict[str, int],
#         dependency_key: Dict[str, int],
#         mapping: List[Tuple[str, List[str]]],
#         device=torch.device('cpu')
#     ) -> Tuple[torch.tensor, Tuple[Dict, Dict]]:
#     """Create a sparse binary mapping tensor from targets to dependencies.

#     Args:
#         target_key (dict of str): list of target names corresponding to
#             the indices in the target tensor.
#         dependency_key (dict of str): list of dependency names corresponding to
#             the indices in the dependency tensor.
#         mapping (dict): mapping from target names to lists of dependency
#             names they depend on.

#     Returns:
#         torch.tensor: sparse binary mapping of target to set of
#             dependencies. (t, d)
#         tuple: (target_key, dependency_key)
#     """

#     rows, cols = [], []
#     for target, dependencies in mapping:
#         if target not in target_key:
#             raise ValueError(
#                 f"Target {target} not found in target_key."
#             )
#         t_idx = target_key[target]
#         for dep in dependencies:
#             if dep not in dependency_key:
#                 raise ValueError(
#                     f"Dependency {dep} not found in dependency_key."
#                 )
#             d_idx = dependency_key[dep]
#             rows.append(t_idx)
#             cols.append(d_idx)

#     all_indicies = torch.tensor([rows, cols], device=device)
#     values = torch.ones(len(rows), device=device)
#     mapping_tensor = torch.sparse_coo_tensor(
#         all_indicies, 
#         values, 
#         (len(target_key), len(dependency_key)),
#         device=device
#     )

#     return mapping_tensor, (target_key, dependency_key)


# def calc_modifiers_to_dependencies(
#         modifiers: torch.tensor, 
#         modifier_key: Dict[str, int], 
#         dependency_key: Dict[str, int], 
#         mapping: List[Tuple[str, str, str]]):
#     """Convert of modifier values to a vector of dependencies.
    
#     Args:
#         modifiers (torch.tensor): scalar values for each modifier m, 
#             representing its influence on the body mesh. (m,) Should be full
#             of values (including default values).
#         modifier_key (dict of str): list of modifier names corresponding to
#             the indices in the modifier tensor.
#         dependency_key (list of str): list of dependency names to output.
#         mapping (dict): mapping from modifier names to lists of dependency
#             names they influence and the lambda function associated with them.
#             The lambda function should use torch functionality only.

#     Returns:
#         torch.tensor: scalar values for each dependency d as a function of 
#             the modifier values. In order of the dependency_key
#     """
#     dependencies = torch.ones(len(dependency_key), device=modifiers.device)

#     for dep, mod, function in mapping:
#         if mod not in modifier_key:
#             raise ValueError(
#                 f"Modifier {mod} not found in modifier_key."
#             )
#         if dep not in dependency_key:
#             raise ValueError(
#                 f"Dependency {dep} not found in dependency_key."
#             )

#         i_d, i_m = dependency_key[dep], modifier_key[mod]

#         dependencies[i_d] = eval(function)(modifiers[i_m])

#     return dependencies, dependency_key

# def calc_dependencies_to_targets(
#         dependencies: torch.tensor, 
#         dependencies_key: Dict[str, int],
#         mapping: torch.tensor,
#         mapping_keys: Tuple[Dict, Dict]):
#     """Convert of dependency values to a vector of targets.
    
#     Args:
#         dependencies (torch.tensor): scalar values for each dependency d. (d,)
#         mapping_tensor (torch.tensor): sparse binary mapping of target to set of 
#             dependencies. (t, d)
#         mapping_keys (tuple): (target_key, dependency_key) dictionaries.

#     Returns:
#         torch.tensor: scalar values for each target t as a function of 
#             the dependency values. In order of the target_key (t,)
#     """
#     log_dependencies = torch.log(dependencies)
#     targets_log = torch.sparse.mm(mapping, log_dependencies) # assumed to handle -inf well...
#     targets = torch.exp(targets_log)

#     return targets