import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import math


class SwinTransformerPatchEmbedding(nn.Module):
    """Simplified Swin Transformer patch embedding for feature extraction"""

    def __init__(self, patch_size: int = 4, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return x, (H, W)


class GridGraphFeature(nn.Module):
    """
    Grid-graph feature extraction based on the paper description
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        gcn_hidden_dim: int = 128,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.gcn_hidden_dim = gcn_hidden_dim

        # Calculate number of patches
        self.num_patches_per_side = image_size // patch_size  # M = k = 56 for 224/4
        self.num_patches = self.num_patches_per_side**2  # M*k patches total

        # Swin Transformer for patch feature extraction
        self.swin_embedding = SwinTransformerPatchEmbedding(patch_size, embed_dim)

        # Graph Convolutional Layer
        self.gcn = GraphConvolutionalLayer(embed_dim, gcn_hidden_dim)

    def extract_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features using Swin Transformer
        Args:
            image: [B, 3, 224, 224] input image
        Returns:
            X^V: [B, M*k, embed_dim] patch feature matrix
        """
        # Split image into M x k patches and extract features
        patch_features, (H, W) = self.swin_embedding(image)  # [B, M*k, embed_dim]
        return patch_features

    def compute_similarity_matrix(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between patches
        Args:
            patch_features: [B, M*k, embed_dim] patch features
        Returns:
            A^V: [B, M*k, M*k] adjacency matrix
        """
        B, N, D = patch_features.shape
        k = self.num_patches_per_side

        # Normalize patch features for cosine similarity
        patch_features_norm = F.normalize(patch_features, p=2, dim=-1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.bmm(
            patch_features_norm, patch_features_norm.transpose(-2, -1)
        )

        # Initialize adjacency matrix
        adjacency_matrix = torch.zeros_like(similarity_matrix)

        # Create grid indices
        indices = torch.arange(N, device=patch_features.device).reshape(k, k)

        # Horizontal neighbors (right and left)
        right_neighbors = indices[:, :-1].flatten()  # i
        right_targets = indices[:, 1:].flatten()  # j
        left_neighbors = right_targets  # i
        left_targets = right_neighbors  # j

        # Vertical neighbors (down and up)
        down_neighbors = indices[:-1, :].flatten()  # i
        down_targets = indices[1:, :].flatten()  # j
        up_neighbors = down_targets  # i
        up_targets = down_neighbors  # j

        # Set adjacency for neighbors to 1
        adjacency_matrix[:, right_neighbors, right_targets] = 1.0
        adjacency_matrix[:, left_neighbors, left_targets] = 1.0
        adjacency_matrix[:, down_neighbors, down_targets] = 1.0
        adjacency_matrix[:, up_neighbors, up_targets] = 1.0

        # Set diagonal to 1
        adjacency_matrix[:, torch.arange(N), torch.arange(N)] = 1.0

        # Fill non-adjacent positions with similarity values
        mask = adjacency_matrix == 0
        adjacency_matrix[mask] = similarity_matrix[mask]

        return adjacency_matrix

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of grid-graph feature extraction
        Args:
            image: [B, 3, 224, 224] input image
        Returns:
            V^(0): [B, M*k, hidden_dim] initial context grid representation
        """
        # Step 1: Extract patch features using Swin Transformer
        patch_features = self.extract_patch_features(image)  # X^V

        # Step 2: Compute similarity-based adjacency matrix
        adjacency_matrix = self.compute_similarity_matrix(patch_features)  # A^V

        # Step 3: Apply Graph Convolutional Layer
        grid_representation = self.gcn(patch_features, adjacency_matrix)  # V^(0)

        return grid_representation


class GraphConvolutionalLayer(nn.Module):
    """
    Graph Convolutional Layer implementation
    V^(0) = σ(Ã^T VW^T)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Weight matrix W
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, input_dim] node feature matrix V
            adjacency_matrix: [B, N, N] adjacency matrix A^V
        Returns:
            output: [B, N, output_dim] output features V^(0)
        """
        # Compute degree matrix D
        degree_matrix = torch.sum(adjacency_matrix, dim=-1, keepdim=True)  # [B, N, 1]
        degree_matrix = torch.clamp(degree_matrix, min=1.0)  # Avoid division by zero

        # Normalize adjacency matrix: Ã = D^(-1/2) A^V D^(-1/2)
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5)
        normalized_adj = (
            degree_inv_sqrt * adjacency_matrix * degree_inv_sqrt.transpose(-2, -1)
        )

        # Apply weight transformation: VW^T
        transformed_features = torch.matmul(
            node_features, self.weight
        )  # [B, N, output_dim]

        # Graph convolution: Ã^T VW^T
        output = torch.bmm(normalized_adj.transpose(-2, -1), transformed_features)

        # Apply activation function (ReLU)
        output = F.relu(output)

        return output


def test_grid_graph_feature():
    """Test the Grid-Graph Feature implementation"""

    # Create model
    model = GridGraphFeature(
        image_size=224, patch_size=4, embed_dim=96, gcn_hidden_dim=128
    )

    # Create dummy input
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)

    print(f"Input image shape: {dummy_image.shape}")
    print(f"Number of patches: {model.num_patches}")
    print(f"Patches per side: {model.num_patches_per_side}")

    # Forward pass
    with torch.no_grad():
        grid_representation = model(dummy_image)

    print(f"Output grid representation shape: {grid_representation.shape}")
    print(
        f"Expected shape: [{batch_size}, {model.num_patches}, {model.gcn_hidden_dim}]"
    )

    # Test individual components
    print("\n--- Testing individual components ---")

    # Test patch feature extraction
    patch_features = model.extract_patch_features(dummy_image)
    print(f"Patch features shape: {patch_features.shape}")

    # Test similarity matrix computation
    similarity_matrix = model.compute_similarity_matrix(patch_features)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Verify adjacency matrix properties
    print(
        f"Adjacency matrix diagonal sum: {torch.diagonal(similarity_matrix, dim1=-2, dim2=-1).sum()}"
    )
    print(
        f"Adjacency matrix range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]"
    )


if __name__ == "__main__":
    test_grid_graph_feature()
