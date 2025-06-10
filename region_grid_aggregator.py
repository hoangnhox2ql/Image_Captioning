import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data, Batch


class RegionGridAggregator(nn.Module):
    def __init__(
        self,
        grid_feature_dim: int = 128,
        region_feature_dim: int = 512,
        hidden_dim: int = 256,
        num_cluster_centers: int = 10,
        gnn_type: str = "transformer",  # 'gcn', 'gat', 'transformer'
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.grid_feature_dim = grid_feature_dim
        self.region_feature_dim = region_feature_dim
        self.hidden_dim = hidden_dim
        self.num_cluster_centers = num_cluster_centers
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers

        # Feature projection layers
        self.grid_proj = nn.Linear(grid_feature_dim, hidden_dim)
        self.region_proj = nn.Linear(region_feature_dim, hidden_dim)

        # Learnable cluster centers (regional centers r_i^(0))
        self.cluster_centers = nn.Parameter(
            torch.randn(num_cluster_centers, hidden_dim)
        )

        # Correlation computation parameters (b_i and b_j in equation 9)
        self.grid_bias = nn.Parameter(torch.zeros(1))
        self.region_bias = nn.Parameter(torch.zeros(1))

        # Learnable parameter r̄ (magnitude equal to r_i^(0))
        self.learnable_r_bar = nn.Parameter(torch.randn(hidden_dim))

        # GNN layers for graph-based feature aggregation
        self.gnn_layers = nn.ModuleList()

        if gnn_type == "gcn":
            for i in range(num_gnn_layers):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == "gat":
            for i in range(num_gnn_layers):
                self.gnn_layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
        elif gnn_type == "transformer":
            for i in range(num_gnn_layers):
                self.gnn_layers.append(
                    TransformerConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )

        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # Final output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters"""
        # Initialize cluster centers with Xavier uniform
        nn.init.xavier_uniform_(self.cluster_centers)
        nn.init.xavier_uniform_(self.learnable_r_bar.unsqueeze(0))

        # Initialize bias parameters
        nn.init.zeros_(self.grid_bias)
        nn.init.zeros_(self.region_bias)

    def compute_correlation_weights(
        self, grid_features: torch.Tensor, region_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation weights r_{j,i}^(0) between grid features and region centers
        Following equation (9) in the paper

        Args:
            grid_features: [B, M, hidden_dim] projected grid features v_j^(0)
            region_centers: [B, K, hidden_dim] region centers r_i^(0)
        Returns:
            correlation_weights: [B, M, K] correlation weights r_{j,i}^(0)
        """
        B, M, D = grid_features.shape
        K = region_centers.shape[1]

        # Expand dimensions for broadcasting
        grid_expanded = grid_features.unsqueeze(2)  # [B, M, 1, D]
        centers_expanded = region_centers.unsqueeze(1)  # [B, 1, K, D]

        # Compute dot product v_j^(0) · r_i^(0)
        dot_products = torch.sum(grid_expanded * centers_expanded, dim=-1)  # [B, M, K]

        # Add bias terms (b_i and b_j)
        dot_products = dot_products + self.grid_bias + self.region_bias

        # Apply softmax to get correlation weights (equation 9)
        correlation_weights = F.softmax(dot_products, dim=-1)  # [B, M, K]

        return correlation_weights

    def aggregate_region_features(
        self, grid_features: torch.Tensor, correlation_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate grid features into region features using correlation weights
        Following equation (10) in the paper

        Args:
            grid_features: [B, M, hidden_dim] projected grid features
            correlation_weights: [B, M, K] correlation weights
        Returns:
            aggregated_regions: [B, K, hidden_dim] aggregated region features r_i^(1)
        """
        B, M, D = grid_features.shape
        K = correlation_weights.shape[-1]

        # Weighted sum: Σ(r_{j,i}^(0) * (v_j^(0) - r̄))
        r_bar_expanded = self.learnable_r_bar.unsqueeze(0).unsqueeze(0).expand(B, M, -1)
        grid_centered = grid_features - r_bar_expanded  # v_j^(0) - r̄

        # Compute weighted sum
        weighted_features = correlation_weights.unsqueeze(-1) * grid_centered.unsqueeze(
            2
        )  # [B, M, K, D]
        aggregated_features = torch.sum(weighted_features, dim=1)  # [B, K, D]

        # Apply L2 normalization (Norm in equation 10)
        aggregated_regions = F.normalize(aggregated_features, p=2, dim=-1)

        return aggregated_regions

    def build_graph(
        self,
        grid_features: torch.Tensor,
        region_features: torch.Tensor,
        aggregated_regions: torch.Tensor,
    ) -> List[Data]:
        """
        Build graph structure for GNN processing
        """
        B, M, D = grid_features.shape
        P = region_features.shape[1]
        K = aggregated_regions.shape[1]
        device = grid_features.device

        graph_list = []

        for b in range(B):
            node_features = torch.cat(
                [grid_features[b], region_features[b], aggregated_regions[b]], dim=0
            )

            edge_indices = []

            # 1. Grid-to-Grid connections
            grid_edges = self._build_grid_edges(M, device)
            edge_indices.append(grid_edges)

            # 2. Region-to-Region connections
            region_edges = self._build_region_edges(region_features[b], M, P)
            edge_indices.append(region_edges)

            # 3. Grid-to-Aggregated connections
            grid_agg_edges = self._build_grid_aggregated_edges(
                grid_features[b], aggregated_regions[b], M, P, K
            )
            edge_indices.append(grid_agg_edges)

            # 4. Region-to-Aggregated connections
            region_agg_edges = self._build_region_aggregated_edges(
                region_features[b], aggregated_regions[b], M, P, K
            )
            edge_indices.append(region_agg_edges)

            # Combine all edges
            all_edge_indices = torch.cat(edge_indices, dim=1)

            graph_data = Data(x=node_features, edge_index=all_edge_indices)

            graph_list.append(graph_data)

        return graph_list

    def _build_grid_edges(self, M: int, device: torch.device) -> torch.Tensor:
        """Build spatial connections between grid nodes"""
        grid_size = int(math.sqrt(M))
        edge_indices = []

        for i in range(grid_size):
            for j in range(grid_size):
                current_idx = i * grid_size + j
                if j < grid_size - 1:
                    right_idx = i * grid_size + (j + 1)
                    edge_indices.extend(
                        [[current_idx, right_idx], [right_idx, current_idx]]
                    )
                if i < grid_size - 1:
                    bottom_idx = (i + 1) * grid_size + j
                    edge_indices.extend(
                        [[current_idx, bottom_idx], [bottom_idx, current_idx]]
                    )

        if len(edge_indices) > 0:
            return (
                torch.tensor(edge_indices, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

    def _build_region_edges(
        self, region_features: torch.Tensor, M: int, P: int
    ) -> torch.Tensor:
        """Build semantic connections between region nodes"""
        device = region_features.device

        region_norm = F.normalize(region_features, p=2, dim=-1)
        similarity_matrix = torch.mm(region_norm, region_norm.t())

        k = min(5, P - 1 if P > 1 else 0)
        if k == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_indices = []

        for i in range(P):
            similarities = similarity_matrix[i]
            similarities[i] = -1
            _, top_indices = torch.topk(similarities, k)

            for j in top_indices:
                if similarities[j] > 0.5:
                    edge_indices.extend([[M + i, M + j], [M + j, M + i]])

        if len(edge_indices) > 0:
            return (
                torch.tensor(edge_indices, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

    def _build_grid_aggregated_edges(
        self,
        grid_features: torch.Tensor,
        aggregated_regions: torch.Tensor,
        M: int,
        P: int,
        K: int,
    ) -> torch.Tensor:
        """Build correlation-based connections between grid and aggregated nodes"""
        device = grid_features.device

        correlation_weights = self.compute_correlation_weights(
            grid_features.unsqueeze(0), aggregated_regions.unsqueeze(0)
        ).squeeze(0)

        edge_indices = []
        top_k = min(3, K)
        if top_k == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        for i in range(M):
            _, top_indices = torch.topk(correlation_weights[i], top_k)
            for j in top_indices:
                if correlation_weights[i, j].item() > 0.1:
                    edge_indices.extend([[i, M + P + j], [M + P + j, i]])

        if len(edge_indices) > 0:
            return (
                torch.tensor(edge_indices, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

    def _build_region_aggregated_edges(
        self,
        region_features: torch.Tensor,
        aggregated_regions: torch.Tensor,
        M: int,
        P: int,
        K: int,
    ) -> torch.Tensor:
        """Build attention-based connections between region and aggregated nodes"""
        device = region_features.device

        region_norm = F.normalize(region_features, p=2, dim=-1)
        aggregated_norm = F.normalize(aggregated_regions, p=2, dim=-1)
        attention_scores = torch.mm(region_norm, aggregated_norm.t())

        edge_indices = []
        top_k = min(2, K)
        if top_k == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        for i in range(P):
            _, top_indices = torch.topk(attention_scores[i], top_k)
            for j in top_indices:
                if attention_scores[i, j].item() > 0.3:
                    edge_indices.extend([[M + i, M + P + j], [M + P + j, M + i]])

        if len(edge_indices) > 0:
            return (
                torch.tensor(edge_indices, dtype=torch.long, device=device)
                .t()
                .contiguous()
            )
        else:
            return torch.empty((2, 0), dtype=torch.long, device=device)

    def forward(
        self, grid_features: torch.Tensor, region_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, M, _ = grid_features.shape
        P = region_features.shape[1]
        K = self.num_cluster_centers

        grid_projected = self.grid_proj(grid_features)
        region_projected = self.region_proj(region_features)

        cluster_centers_batch = self.cluster_centers.unsqueeze(0).expand(B, -1, -1)

        correlation_weights = self.compute_correlation_weights(
            grid_projected, cluster_centers_batch
        )
        aggregated_regions = self.aggregate_region_features(
            grid_projected, correlation_weights
        )

        graph_list = self.build_graph(
            grid_projected, region_projected, aggregated_regions
        )
        graph_batch = Batch.from_data_list(graph_list)

        x, edge_index = graph_batch.x, graph_batch.edge_index

        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == "gcn":
                x_new = gnn_layer(x, edge_index)
            else:
                x_new = gnn_layer(x, edge_index)

            x = self.layer_norms[i](x + x_new)
            x = self.dropout(x)

        x = self.output_proj(x)

        node_counts = [M + P + K] * B
        feature_list = x.split(node_counts)

        enhanced_grid_features_list = []
        enhanced_region_features_list = []
        enhanced_aggregated_regions_list = []

        for features in feature_list:
            enhanced_grid_features_list.append(features[:M])
            enhanced_region_features_list.append(features[M : M + P])
            enhanced_aggregated_regions_list.append(features[M + P :])

        enhanced_grid_features = torch.stack(enhanced_grid_features_list)
        enhanced_region_features = torch.stack(enhanced_region_features_list)
        enhanced_aggregated_regions = torch.stack(enhanced_aggregated_regions_list)

        return (
            enhanced_grid_features,
            enhanced_region_features,
            enhanced_aggregated_regions,
        )


class MMGATWithRegionGridAggregator(nn.Module):
    """Complete MMGAT model with Region-Grid Aggregator"""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        grid_embed_dim: int = 96,
        region_feature_dim: int = 512,
        hidden_dim: int = 256,
        num_regions: int = 50,
        num_cluster_centers: int = 10,
        gnn_type: str = "transformer",
    ):
        super().__init__()

        # Region-Grid Aggregator
        self.region_grid_aggregator = RegionGridAggregator(
            grid_feature_dim=128,  # Output from GridGraphFeature
            region_feature_dim=region_feature_dim,
            hidden_dim=hidden_dim,
            num_cluster_centers=num_cluster_centers,
            gnn_type=gnn_type,
        )

        # Final fusion layer
        self.final_fusion = nn.Linear(
            hidden_dim * 3, hidden_dim
        )  # Grid + Region + Aggregated

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass
        Args:
            images: [B, 3, H, W] input images
        Returns:
            fused_features: [B, N_total, hidden_dim] final fused features
        """
        # For demonstration, create dummy features
        B = images.shape[0]
        M = (224 // 4) ** 2  # Grid patches
        P = 50  # Number of regions

        grid_features = torch.randn(B, M, 128, device=images.device)
        region_features = torch.randn(B, P, 512, device=images.device)

        # Apply Region-Grid Aggregator
        enhanced_grid, enhanced_region, aggregated_region = self.region_grid_aggregator(
            grid_features, region_features
        )

        # Combine all features
        combined_features = torch.cat(
            [
                enhanced_grid,  # [B, M, hidden_dim]
                enhanced_region,  # [B, P, hidden_dim]
                aggregated_region,  # [B, K, hidden_dim]
            ],
            dim=1,
        )  # [B, M+P+K, hidden_dim]

        return combined_features


def test_region_grid_aggregator():
    """Test the Region-Grid Aggregator implementation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = RegionGridAggregator(
        grid_feature_dim=128,
        region_feature_dim=512,
        hidden_dim=256,
        num_cluster_centers=10,
        gnn_type="transformer",
        num_gnn_layers=3,
    ).to(device)

    # Create dummy input features
    batch_size = 2
    M = 56 * 56  # Grid features (56x56 grid)
    P = 50  # Region features

    grid_features = torch.randn(batch_size, M, 128).to(device)
    region_features = torch.randn(batch_size, P, 512).to(device)

    print(f"Input grid features shape: {grid_features.shape}")
    print(f"Input region features shape: {region_features.shape}")

    # Forward pass
    with torch.no_grad():
        enhanced_grid, enhanced_region, aggregated_region = model(
            grid_features, region_features
        )

    print(f"Enhanced grid features shape: {enhanced_grid.shape}")
    print(f"Enhanced region features shape: {enhanced_region.shape}")
    print(f"Aggregated region features shape: {aggregated_region.shape}")

    # Test correlation weights computation
    grid_proj = model.grid_proj(grid_features)
    cluster_centers_batch = model.cluster_centers.unsqueeze(0).expand(
        batch_size, -1, -1
    )
    correlation_weights = model.compute_correlation_weights(
        grid_proj, cluster_centers_batch
    )
    print(f"Correlation weights shape: {correlation_weights.shape}")
    print(
        f"Correlation weights sum (should be ~1.0): {correlation_weights.sum(dim=-1).mean():.3f}"
    )

    if device.type == "cuda":
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
        )


if __name__ == "__main__":
    test_region_grid_aggregator()
