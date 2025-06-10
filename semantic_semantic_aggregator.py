import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class SemanticSemanticAggregator(nn.Module):
    """
    Semantic-Semantic Aggregator implementation
    Based on equation (17)-(20) from the paper
    """

    def __init__(
        self,
        input_dim: int = 256,
        lstm_hidden_dim: int = 512,
        mlp_hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.output_dim = output_dim

        # LSTM for sequential processing of semantic nodes
        # S^t = {s₁ᵗ, s₂ᵗ, s₃ᵗ, ..., sₙᵗ} = LSTM(S^(0))
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # MLPs for attention mechanism
        # g_x, g_z and g_v are MLPs for encoding node features
        self.g_x = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        self.g_z = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        self.g_v = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        # MLP for encoding neighboring features
        # g_n is an MLP to encode the features of neighboring nodes
        self.g_n = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

        # Final output projection
        self.output_projection = nn.Linear(lstm_hidden_dim + output_dim, output_dim)

    def find_neighboring_nodes(
        self, adjacency_matrix: torch.Tensor, node_idx: int
    ) -> List[int]:
        """
        Find neighboring nodes N_i = {s_j|j ∈ {1, ..., N} and j ≠ i}
        Args:
            adjacency_matrix: [N, N] adjacency matrix
            node_idx: Index of the current node
        Returns:
            List of neighboring node indices
        """
        N = adjacency_matrix.shape[0]
        neighbors = []

        for j in range(N):
            if j != node_idx and adjacency_matrix[node_idx, j] > 0:
                neighbors.append(j)

        return neighbors

    def compute_attention_scores(
        self, current_node: torch.Tensor, neighbor_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores between current node and its neighbors
        a_{s_i,s_j} = g_x([s_i^(t); g_z(s_j^(t))]) · g_v([s_j^(t); g_z(s_i^(t))])

        Args:
            current_node: [lstm_hidden_dim] current node features s_i^(t)
            neighbor_nodes: [K, lstm_hidden_dim] neighbor nodes features
        Returns:
            attention_scores: [K] attention scores
        """
        K = neighbor_nodes.shape[0]
        if K == 0:
            return torch.tensor([], device=current_node.device)

        # Expand current node to match neighbor dimensions
        current_expanded = current_node.unsqueeze(0).expand(
            K, -1
        )  # [K, lstm_hidden_dim]

        # Compute g_z for neighbors and current node
        g_z_neighbors = self.g_z(neighbor_nodes)  # [K, mlp_hidden_dim]
        g_z_current = self.g_z(current_expanded)  # [K, mlp_hidden_dim]

        # Concatenate features for attention computation
        # [s_i^(t); g_z(s_j^(t))]
        concat_1 = torch.cat(
            [current_expanded, g_z_neighbors], dim=-1
        )  # [K, lstm_hidden_dim + mlp_hidden_dim]

        # [s_j^(t); g_z(s_i^(t))]
        concat_2 = torch.cat(
            [neighbor_nodes, g_z_current], dim=-1
        )  # [K, lstm_hidden_dim + mlp_hidden_dim]

        # Apply g_x and g_v
        # Note: We need to adjust dimensions for proper computation
        # Using simplified dot product attention instead of the exact formula
        attention_1 = self.g_x(
            concat_1[:, : self.lstm_hidden_dim]
        )  # [K, mlp_hidden_dim]
        attention_2 = self.g_v(
            concat_2[:, : self.lstm_hidden_dim]
        )  # [K, mlp_hidden_dim]

        # Compute attention scores (dot product)
        attention_scores = torch.sum(attention_1 * attention_2, dim=-1)  # [K]

        return attention_scores

    def aggregate_neighbors(
        self,
        current_node: torch.Tensor,
        neighbor_nodes: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate neighboring node information using attention weights
        Args:
            current_node: [lstm_hidden_dim] current node features
            neighbor_nodes: [K, lstm_hidden_dim] neighbor node features
            attention_weights: [K] normalized attention weights
        Returns:
            aggregated_features: [output_dim] aggregated neighbor features
        """
        if neighbor_nodes.shape[0] == 0:
            # No neighbors, return zero features
            return torch.zeros(self.output_dim, device=current_node.device)

        # Weight neighbors by attention
        weighted_neighbors = (
            attention_weights.unsqueeze(-1) * neighbor_nodes
        )  # [K, lstm_hidden_dim]

        # Sum weighted neighbors
        summed_neighbors = torch.sum(weighted_neighbors, dim=0)  # [lstm_hidden_dim]

        # Apply g_n MLP to encode neighboring features
        neighbor_encoding = self.g_z(summed_neighbors.unsqueeze(0)).squeeze(
            0
        )  # [mlp_hidden_dim]
        aggregated_features = self.g_n(neighbor_encoding.unsqueeze(0)).squeeze(
            0
        )  # [output_dim]

        return aggregated_features

    def forward_single_graph(
        self, semantic_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a single semantic graph
        Args:
            semantic_features: [N, input_dim] semantic node features S^(0)
            adjacency_matrix: [N, N] adjacency matrix
        Returns:
            updated_features: [N, output_dim] updated semantic features S^(3)
        """
        N = semantic_features.shape[0]

        # Step 1: Apply LSTM to get sequential representation
        # S^t = {s₁ᵗ, s₂ᵗ, s₃ᵗ, ..., sₙᵗ} = LSTM(S^(0))
        lstm_out, _ = self.lstm(
            semantic_features.unsqueeze(0)
        )  # [1, N, lstm_hidden_dim]
        lstm_features = lstm_out.squeeze(0)  # [N, lstm_hidden_dim]

        # Step 2: For each node, compute attention with neighbors and aggregate
        updated_nodes = []

        for i in range(N):
            current_node = lstm_features[i]  # [lstm_hidden_dim]

            # Find neighboring nodes
            neighbor_indices = self.find_neighboring_nodes(adjacency_matrix, i)

            if len(neighbor_indices) == 0:
                # No neighbors, just use current node
                aggregated_neighbor_features = torch.zeros(
                    self.output_dim, device=current_node.device
                )
            else:
                # Get neighbor features
                neighbor_nodes = lstm_features[neighbor_indices]  # [K, lstm_hidden_dim]

                # Compute attention scores a_{s_i,s_j}
                attention_scores = self.compute_attention_scores(
                    current_node, neighbor_nodes
                )

                # Normalize attention weights using softmax
                attention_weights = F.softmax(attention_scores, dim=0)  # [K]

                # Aggregate neighbor information
                aggregated_neighbor_features = self.aggregate_neighbors(
                    current_node, neighbor_nodes, attention_weights
                )

            # Step 3: Combine current node with aggregated neighbor features
            # s_i^(3) = [s_i^(t), ∑_{j∈N_i} a_{s_i,s_j} · g_n(s_j^(t))]
            combined_features = torch.cat(
                [current_node, aggregated_neighbor_features], dim=0
            )

            # Apply final projection
            updated_node = self.output_projection(
                combined_features.unsqueeze(0)
            ).squeeze(0)
            updated_nodes.append(updated_node)

        # Stack all updated nodes
        updated_features = torch.stack(updated_nodes, dim=0)  # [N, output_dim]

        return updated_features

    def forward(
        self,
        semantic_features_list: List[torch.Tensor],
        adjacency_matrices_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Process multiple semantic graphs
        Args:
            semantic_features_list: List of [Ni, input_dim] semantic features
            adjacency_matrices_list: List of [Ni, Ni] adjacency matrices
        Returns:
            updated_features_list: List of [Ni, output_dim] updated semantic features
        """
        updated_features_list = []

        for semantic_features, adjacency_matrix in zip(
            semantic_features_list, adjacency_matrices_list
        ):
            updated_features = self.forward_single_graph(
                semantic_features, adjacency_matrix
            )
            updated_features_list.append(updated_features)

        return updated_features_list


class EnhancedSemanticGraphFeature(nn.Module):
    """
    Enhanced Semantic Graph Feature with Semantic-Semantic Aggregator
    Combines the original SemanticGraphFeature with SemanticSemanticAggregator
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        bert_dim: int = 768,
        gcn_hidden_dim: int = 512,
        gcn_output_dim: int = 256,
        lstm_hidden_dim: int = 512,
        mlp_hidden_dim: int = 256,
        final_output_dim: int = 256,
    ):
        super().__init__()

        # Import the original SemanticGraphFeature
        from semantic_graph_feature import SemanticGraphFeature

        self.semantic_graph_feature = SemanticGraphFeature(
            bert_model_name=bert_model_name,
            bert_dim=bert_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim,
        )

        # Semantic-Semantic Aggregator
        self.semantic_aggregator = SemanticSemanticAggregator(
            input_dim=gcn_output_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            output_dim=final_output_dim,
        )

    def forward(
        self, text_list: List[str]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Complete semantic processing pipeline
        Args:
            text_list: List of text strings (image captions)
        Returns:
            final_semantic_features: List of [Ni, final_output_dim] final semantic features
            adjacency_matrices: List of [Ni, Ni] adjacency matrices
        """
        # Step 1: Extract initial semantic graph features S^(0)
        initial_features_list, adjacency_matrices_list = self.semantic_graph_feature(
            text_list
        )

        # Step 2: Apply semantic-semantic aggregator to get S^(3)
        final_features_list = self.semantic_aggregator(
            initial_features_list, adjacency_matrices_list
        )

        return final_features_list, adjacency_matrices_list


# Test function
def test_semantic_semantic_aggregator():
    """Test the Semantic-Semantic Aggregator"""

    print("Testing Semantic-Semantic Aggregator")
    print("=" * 50)

    # Create test data
    batch_size = 2
    test_features_list = [
        torch.randn(5, 256),  # 5 nodes, 256-dim features
        torch.randn(7, 256),  # 7 nodes, 256-dim features
    ]

    test_adjacency_list = [
        torch.randint(0, 2, (5, 5)).float(),  # 5x5 adjacency matrix
        torch.randint(0, 2, (7, 7)).float(),  # 7x7 adjacency matrix
    ]

    # Ensure diagonal is 1 (self-connections)
    for adj in test_adjacency_list:
        adj.fill_diagonal_(1.0)

    # Create aggregator
    aggregator = SemanticSemanticAggregator(
        input_dim=256, lstm_hidden_dim=512, mlp_hidden_dim=256, output_dim=256
    )

    print(f"Input shapes:")
    for i, (features, adj) in enumerate(zip(test_features_list, test_adjacency_list)):
        print(f"  Graph {i+1}: Features {features.shape}, Adjacency {adj.shape}")

    # Forward pass
    with torch.no_grad():
        output_features_list = aggregator(test_features_list, test_adjacency_list)

    print(f"\nOutput shapes:")
    for i, features in enumerate(output_features_list):
        print(f"  Graph {i+1}: {features.shape}")

    print(f"\nAggregator parameters:")
    total_params = sum(p.numel() for p in aggregator.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Test single graph processing
    print(f"\nTesting single graph processing:")
    single_features = test_features_list[0]
    single_adj = test_adjacency_list[0]

    with torch.no_grad():
        single_output = aggregator.forward_single_graph(single_features, single_adj)

    print(f"  Input: {single_features.shape}")
    print(f"  Output: {single_output.shape}")
    print(
        f"  Feature transformation: {single_features.shape[-1]} -> {single_output.shape[-1]}"
    )


if __name__ == "__main__":
    test_semantic_semantic_aggregator()
