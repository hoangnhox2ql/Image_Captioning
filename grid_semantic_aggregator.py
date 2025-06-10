import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict
import numpy as np


class GridSemanticAggregator(nn.Module):
    """
    Grid-Semantic Aggregator that combines grid-graph and semantic-graph features
    Based on the paper's dual objectives:
    1) Refine semantics nodes utilizing the visual context
    2) Enhance visual nodes with contextual semantics
    """

    def __init__(
        self,
        grid_feature_dim: int = 128,  # Output dim from GridGraphFeature
        semantic_feature_dim: int = 256,  # Output dim from SemanticGraphFeature
        lstm_hidden_dim: int = 256,
        mlp_hidden_dim: int = 512,
        final_output_dim: int = 512,
    ):
        super().__init__()

        self.grid_feature_dim = grid_feature_dim
        self.semantic_feature_dim = semantic_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.final_output_dim = final_output_dim

        # Bidirectional LSTM for processing region aggregation features R^(1)
        self.bidirectional_lstm = nn.LSTM(
            input_size=grid_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # MLP layers for encoding different node types
        self.f_s = nn.Sequential(  # For encoding semantic nodes s_i^(0)
            nn.Linear(semantic_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        self.f_c = nn.Sequential(  # For encoding context grid features
            nn.Linear(lstm_hidden_dim * 2, mlp_hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        self.f_v = nn.Sequential(  # For encoding visual series features
            nn.Linear(grid_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        # MLP for correlation score computation
        self.correlation_mlp = nn.Sequential(
            nn.Linear(mlp_hidden_dim * 2, mlp_hidden_dim),  # Concatenated features
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

        # MLPs for adjacent node encoding (f_h for semantic nodes)
        self.f_h = nn.Sequential(  # For encoding adjacent semantic nodes
            nn.Linear(semantic_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        # MLPs for final node encoding (f_g for visual nodes)
        self.f_g = nn.Sequential(  # For encoding s_j^(0) when updating visual nodes
            nn.Linear(semantic_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )

        # Final projection layers
        self.semantic_projection = nn.Linear(
            semantic_feature_dim + mlp_hidden_dim, final_output_dim
        )
        self.visual_projection = nn.Linear(
            grid_feature_dim + mlp_hidden_dim, final_output_dim
        )

    def process_grid_features_with_lstm(
        self, grid_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Process grid features through bidirectional LSTM to get R^(1)
        Args:
            grid_features: [B, M*k, grid_feature_dim] from GridGraphFeature
        Returns:
            lstm_features: [B, M*k, lstm_hidden_dim*2] processed features R^(1)
        """
        # Apply bidirectional LSTM
        lstm_output, _ = self.bidirectional_lstm(
            grid_features
        )  # [B, M*k, lstm_hidden_dim*2]
        return lstm_output

    def compute_correlation_scores(
        self, semantic_features: torch.Tensor, grid_context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation scores α_{s_i,r_j} between semantic and grid features
        Args:
            semantic_features: [N_s, semantic_feature_dim] semantic node features s_i^(0)
            grid_context_features: [M*k, lstm_hidden_dim*2] grid context features from LSTM
        Returns:
            correlation_scores: [N_s, M*k] correlation scores α_{s_i,r_j}
        """
        N_s = semantic_features.shape[0]
        M_k = grid_context_features.shape[0]

        # Encode features
        encoded_semantic = self.f_s(semantic_features)  # [N_s, mlp_hidden_dim]
        encoded_grid = self.f_c(grid_context_features)  # [M*k, mlp_hidden_dim]

        # Compute pairwise correlation scores
        correlation_scores = torch.zeros(N_s, M_k, device=semantic_features.device)

        for i in range(N_s):
            for j in range(M_k):
                # Concatenate features for correlation computation
                concat_features = torch.cat(
                    [encoded_semantic[i : i + 1], encoded_grid[j : j + 1]], dim=1
                )  # [1, mlp_hidden_dim*2]

                # Compute correlation score through MLP
                score = self.correlation_mlp(concat_features)  # [1, 1]
                correlation_scores[i, j] = score.squeeze()

        return correlation_scores

    def apply_attention_aggregation(
        self, correlation_scores: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention-based aggregation using correlation scores
        Args:
            correlation_scores: [N_source, N_target] correlation scores
            target_features: [N_target, feature_dim] features to aggregate
        Returns:
            aggregated_features: [N_source, feature_dim] aggregated features
        """
        # Apply softmax to get attention weights
        attention_weights = F.softmax(correlation_scores, dim=1)  # [N_source, N_target]

        # Weighted aggregation
        aggregated_features = torch.matmul(
            attention_weights, target_features
        )  # [N_source, feature_dim]

        return aggregated_features

    def refine_semantic_nodes(
        self,
        semantic_features: torch.Tensor,
        semantic_adj_matrix: torch.Tensor,
        grid_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine semantic nodes using visual context (Objective 1)
        Implements equations (11)-(14) from the paper
        Args:
            semantic_features: [N_s, semantic_feature_dim] semantic node features S^(0)
            semantic_adj_matrix: [N_s, N_s] semantic adjacency matrix A^S
            grid_features: [M*k, grid_feature_dim] grid features V^(0)
        Returns:
            refined_semantic: [N_s, final_output_dim] refined semantic features S^(2)
        """
        # Step 1: Process grid features through LSTM to get R^(1)
        grid_features_expanded = grid_features.unsqueeze(0)  # Add batch dim
        lstm_features = self.process_grid_features_with_lstm(
            grid_features_expanded
        )  # [1, M*k, lstm_hidden_dim*2]
        lstm_features = lstm_features.squeeze(
            0
        )  # Remove batch dim: [M*k, lstm_hidden_dim*2]

        # Step 2: Compute correlation scores α_{s_i,r_j}
        correlation_scores = self.compute_correlation_scores(
            semantic_features, lstm_features
        )  # [N_s, M*k]

        # Step 3: Aggregate visual context for each semantic node
        visual_context = self.apply_attention_aggregation(
            correlation_scores, lstm_features
        )  # [N_s, lstm_hidden_dim*2]

        # Step 4: Update semantic representation s_i^(2) = [s_i^(0); Σ α_{s_i,r_j} f_h(r_j^(1))]
        # Encode visual context
        encoded_visual_context = self.f_h(
            torch.cat(
                [
                    visual_context,
                    torch.zeros(
                        visual_context.shape[0],
                        max(0, self.semantic_feature_dim - visual_context.shape[1]),
                        device=visual_context.device,
                    ),
                ],
                dim=1,
            )[:, : self.semantic_feature_dim]
        )  # [N_s, mlp_hidden_dim]

        # Concatenate original semantic features with visual context
        enhanced_semantic = torch.cat(
            [semantic_features, encoded_visual_context], dim=1
        )  # [N_s, semantic_feature_dim + mlp_hidden_dim]

        # Project to final dimension
        refined_semantic = self.semantic_projection(
            enhanced_semantic
        )  # [N_s, final_output_dim]

        return refined_semantic

    def enhance_visual_nodes(
        self,
        grid_features: torch.Tensor,
        semantic_features: torch.Tensor,
        semantic_adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance visual nodes with contextual semantics (Objective 2)
        Implements equations (15)-(16) from the paper
        Args:
            grid_features: [M*k, grid_feature_dim] grid features V^(0)
            semantic_features: [N_s, semantic_feature_dim] semantic node features S^(0)
            semantic_adj_matrix: [N_s, N_s] semantic adjacency matrix A^S
        Returns:
            enhanced_visual: [M*k, final_output_dim] enhanced visual features R^(2)
        """
        M_k = grid_features.shape[0]

        # Step 1: Process grid features to get visual series features
        encoded_visual = self.f_v(grid_features)  # [M*k, mlp_hidden_dim]

        # Step 2: Compute correlation scores α_{r_i,s_j} (reverse direction)
        # We need to compute correlation between each visual node and semantic nodes
        correlation_scores = torch.zeros(
            M_k, semantic_features.shape[0], device=grid_features.device
        )

        for i in range(M_k):
            for j in range(semantic_features.shape[0]):
                # Concatenate features for correlation computation
                concat_features = torch.cat(
                    [encoded_visual[i : i + 1], self.f_g(semantic_features[j : j + 1])],
                    dim=1,
                )  # [1, mlp_hidden_dim*2]

                # Compute correlation score
                score = self.correlation_mlp(concat_features)  # [1, 1]
                correlation_scores[i, j] = score.squeeze()

        # Step 3: Aggregate semantic context for each visual node
        semantic_context = self.apply_attention_aggregation(
            correlation_scores, semantic_features
        )  # [M*k, semantic_feature_dim]

        # Step 4: Update visual representation r_i^(2) = [r_i^(1); Σ α_{r_i,s_j} f_g(s_j^(0))]
        # Encode semantic context
        encoded_semantic_context = self.f_g(semantic_context)  # [M*k, mlp_hidden_dim]

        # Concatenate original grid features with semantic context
        enhanced_visual_features = torch.cat(
            [grid_features, encoded_semantic_context], dim=1
        )  # [M*k, grid_feature_dim + mlp_hidden_dim]

        # Project to final dimension
        enhanced_visual = self.visual_projection(
            enhanced_visual_features
        )  # [M*k, final_output_dim]

        return enhanced_visual

    def forward(
        self,
        grid_features: torch.Tensor,
        semantic_features_list: List[torch.Tensor],
        semantic_adj_matrices_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of Grid-Semantic Aggregator
        Args:
            grid_features: [B, M*k, grid_feature_dim] grid features from GridGraphFeature
            semantic_features_list: List of [N_i, semantic_feature_dim] semantic features
            semantic_adj_matrices_list: List of [N_i, N_i] semantic adjacency matrices
        Returns:
            refined_semantic_list: List of refined semantic features
            enhanced_visual: [B, M*k, final_output_dim] enhanced visual features
        """
        batch_size = grid_features.shape[0]
        refined_semantic_list = []
        enhanced_visual_list = []

        for b in range(batch_size):
            # Process each sample in the batch
            current_grid = grid_features[b]  # [M*k, grid_feature_dim]

            # Handle multiple semantic graphs per image (if any)
            if b < len(semantic_features_list):
                current_semantic = semantic_features_list[
                    b
                ]  # [N_s, semantic_feature_dim]
                current_semantic_adj = semantic_adj_matrices_list[b]  # [N_s, N_s]

                # Refine semantic nodes using visual context
                refined_semantic = self.refine_semantic_nodes(
                    current_semantic, current_semantic_adj, current_grid
                )
                refined_semantic_list.append(refined_semantic)

                # Enhance visual nodes with semantic context
                enhanced_visual = self.enhance_visual_nodes(
                    current_grid, current_semantic, current_semantic_adj
                )
                enhanced_visual_list.append(enhanced_visual)

        # Stack enhanced visual features
        if enhanced_visual_list:
            enhanced_visual_tensor = torch.stack(
                enhanced_visual_list, dim=0
            )  # [B, M*k, final_output_dim]
        else:
            # Fallback if no semantic features available
            enhanced_visual_tensor = grid_features

        return refined_semantic_list, enhanced_visual_tensor


class MMGATImageCaptioning(nn.Module):
    """
    Complete MMGAT Image Captioning model that integrates Grid and Semantic features
    """

    def __init__(
        self, grid_graph_model, semantic_graph_model, aggregator_config: Dict = None
    ):  # GridGraphFeature instance, SemanticGraphFeature instance
        super().__init__()

        self.grid_graph_model = grid_graph_model
        self.semantic_graph_model = semantic_graph_model

        # Default aggregator configuration
        if aggregator_config is None:
            aggregator_config = {
                "grid_feature_dim": 128,
                "semantic_feature_dim": 256,
                "lstm_hidden_dim": 256,
                "mlp_hidden_dim": 512,
                "final_output_dim": 512,
            }

        self.grid_semantic_aggregator = GridSemanticAggregator(**aggregator_config)

    def forward(
        self, images: torch.Tensor, captions: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Complete forward pass
        Args:
            images: [B, 3, 224, 224] input images
            captions: List of caption strings for semantic graph construction
        Returns:
            refined_semantic_features: List of refined semantic features
            enhanced_visual_features: [B, M*k, final_output_dim] enhanced visual features
        """
        # Step 1: Extract grid-graph features
        grid_features = self.grid_graph_model(images)  # [B, M*k, grid_feature_dim]

        # Step 2: Extract semantic-graph features
        semantic_features_list, semantic_adj_matrices_list = self.semantic_graph_model(
            captions
        )

        # Step 3: Apply Grid-Semantic Aggregator
        refined_semantic_list, enhanced_visual = self.grid_semantic_aggregator(
            grid_features, semantic_features_list, semantic_adj_matrices_list
        )

        return refined_semantic_list, enhanced_visual


# Example usage and testing
def test_grid_semantic_aggregator():
    """Test the complete Grid-Semantic Aggregator implementation"""

    print("Testing Grid-Semantic Aggregator Implementation")
    print("=" * 60)

    # Import the original models (assuming they're available)
    # from grid_graph_feature import GridGraphFeature
    # from semantic_graph_feature import SemanticGraphFeature

    # Create dummy data for testing
    batch_size = 2
    num_patches = 56 * 56  # M*k = 3136 for 224x224 image with patch_size=4
    grid_feature_dim = 128
    semantic_feature_dim = 256

    # Create dummy grid features
    dummy_grid_features = torch.randn(batch_size, num_patches, grid_feature_dim)

    # Create dummy semantic features (variable length for each sample)
    dummy_semantic_features_list = [
        torch.randn(6, semantic_feature_dim),  # 6 words in first caption
        torch.randn(4, semantic_feature_dim),  # 4 words in second caption
    ]

    # Create dummy semantic adjacency matrices
    dummy_semantic_adj_list = [
        torch.eye(6) + torch.randn(6, 6) * 0.1,  # 6x6 adjacency matrix
        torch.eye(4) + torch.randn(4, 4) * 0.1,  # 4x4 adjacency matrix
    ]

    # Make adjacency matrices symmetric and positive
    for i, adj in enumerate(dummy_semantic_adj_list):
        adj = (adj + adj.T) / 2
        adj = torch.clamp(adj, min=0)
        dummy_semantic_adj_list[i] = adj

    # Create aggregator
    aggregator = GridSemanticAggregator(
        grid_feature_dim=grid_feature_dim,
        semantic_feature_dim=semantic_feature_dim,
        lstm_hidden_dim=256,
        mlp_hidden_dim=512,
        final_output_dim=512,
    )

    print(f"Input shapes:")
    print(f"  Grid features: {dummy_grid_features.shape}")
    print(f"  Semantic features: {[f.shape for f in dummy_semantic_features_list]}")
    print(f"  Semantic adjacency: {[adj.shape for adj in dummy_semantic_adj_list]}")

    # Forward pass
    with torch.no_grad():
        refined_semantic_list, enhanced_visual = aggregator(
            dummy_grid_features,
            dummy_semantic_features_list,
            dummy_semantic_adj_list,
        )

    print(f"\nOutput shapes:")
    print(f"  Refined semantic features: {[f.shape for f in refined_semantic_list]}")
    print(f"  Enhanced visual features: {enhanced_visual.shape}")

    # Test individual components
    print(f"\n" + "=" * 60)
    print("Testing Individual Components")
    print("=" * 60)

    # Test LSTM processing
    lstm_features = aggregator.process_grid_features_with_lstm(dummy_grid_features)
    print(f"LSTM features shape: {lstm_features.shape}")

    # Test correlation computation
    single_semantic = dummy_semantic_features_list[0]
    single_grid = lstm_features[0]  # First sample
    correlation_scores = aggregator.compute_correlation_scores(
        single_semantic, single_grid
    )
    print(f"Correlation scores shape: {correlation_scores.shape}")
    print(
        f"Correlation scores range: [{correlation_scores.min():.3f}, {correlation_scores.max():.3f}]"
    )

    # Test attention aggregation
    aggregated = aggregator.apply_attention_aggregation(correlation_scores, single_grid)
    print(f"Aggregated features shape: {aggregated.shape}")

    print(f"\nTest completed successfully!")


if __name__ == "__main__":
    test_grid_semantic_aggregator()
