import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict


class RegionGridAggregator(nn.Module):
    """
    Region-Grid Aggregator that combines grid features with region features
    to create enhanced region representations
    """

    def __init__(
        self,
        grid_feature_dim: int = 128,
        region_feature_dim: int = 512,
        hidden_dim: int = 256,
        num_regions: int = 100,
    ):
        super().__init__()

        self.grid_feature_dim = grid_feature_dim
        self.region_feature_dim = region_feature_dim
        self.hidden_dim = hidden_dim
        self.num_regions = num_regions

        # Learnable parameters for attention calculation
        self.b_i = nn.Parameter(torch.randn(1))
        self.b_j = nn.Parameter(torch.randn(1))

        # Learnable parameter r_tilde for normalization
        self.r_tilde = nn.Parameter(torch.randn(grid_feature_dim))

        # Projection layer to align dimensions
        self.grid_projection = nn.Linear(grid_feature_dim, region_feature_dim)

    def compute_region_grid_correlation(
        self, grid_features: torch.Tensor, region_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation between region centers and grid features
        Args:
            grid_features: [B, M*k, grid_dim] from grid-graph feature
            region_centers: [B, P, region_dim] potential region centers
        Returns:
            correlation_matrix: [B, P, M*k] correlation scores
        """
        B, num_patches, grid_dim = grid_features.shape
        B, num_regions, region_dim = region_centers.shape

        # Ensure consistent dtype
        grid_features = grid_features.float()
        region_centers = region_centers.float()

        # Project grid features to same dimension as region features
        grid_projected = self.grid_projection(grid_features)  # [B, M*k, region_dim]

        # Compute dot product correlation
        correlation = torch.bmm(
            region_centers, grid_projected.transpose(-2, -1)
        )  # [B, P, M*k]

        # Add learnable bias terms (ensure same dtype)
        correlation = correlation + self.b_i.float() + self.b_j.float()  # Broadcasting

        return correlation

    def compute_attention_weights(
        self, correlation_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights using softmax normalization
        Args:
            correlation_matrix: [B, P, M*k] correlation scores
        Returns:
            attention_weights: [B, P, M*k] normalized attention weights
        """
        # Apply softmax over grid features dimension (M*k)
        attention_weights = F.softmax(correlation_matrix, dim=-1)  # [B, P, M*k]

        return attention_weights

    def aggregate_grid_to_region(
        self, grid_features: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate grid features to region features using attention weights
        Args:
            grid_features: [B, M*k, grid_dim] grid features
            attention_weights: [B, P, M*k] attention weights
        Returns:
            aggregated_features: [B, P, grid_dim] aggregated region features
        """
        # Weighted sum of grid features
        aggregated_features = torch.bmm(
            attention_weights, grid_features
        )  # [B, P, grid_dim]

        return aggregated_features

    def compute_region_aggregation_feature(
        self, aggregated_features: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute final region aggregation feature with L2 normalization
        Args:
            aggregated_features: [B, P, grid_dim] aggregated features
            attention_weights: [B, P, M*k] attention weights
        Returns:
            region_agg_features: [B, P, grid_dim] final region aggregation features
        """
        B, P, grid_dim = aggregated_features.shape

        # Ensure consistent dtype
        aggregated_features = aggregated_features.float()

        # Expand r_tilde for batch processing
        r_tilde_expanded = (
            self.r_tilde.float().unsqueeze(0).unsqueeze(0).expand(B, P, -1)
        )  # [B, P, grid_dim]

        # Compute difference from learnable parameter
        diff = aggregated_features - r_tilde_expanded  # [B, P, grid_dim]

        # Apply L2 norm standardization
        region_agg_features = F.normalize(diff, p=2, dim=-1)  # [B, P, grid_dim]

        return region_agg_features

    def forward(
        self, grid_features: torch.Tensor, region_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of Region-Grid Aggregator
        Args:
            grid_features: [B, M*k, grid_dim] from GridGraphFeature
            region_centers: [B, P, region_dim] potential region centers
        Returns:
            region_aggregation_features: [B, P, grid_dim] enhanced region features
        """
        # Step 1: Compute correlation between regions and grid features
        correlation_matrix = self.compute_region_grid_correlation(
            grid_features, region_centers
        )

        # Step 2: Compute attention weights
        attention_weights = self.compute_attention_weights(correlation_matrix)

        # Step 3: Aggregate grid features to regions
        aggregated_features = self.aggregate_grid_to_region(
            grid_features, attention_weights
        )

        # Step 4: Compute final region aggregation features
        region_aggregation_features = self.compute_region_aggregation_feature(
            aggregated_features, attention_weights
        )

        return region_aggregation_features


class MultiModalGraphAttention(nn.Module):
    """
    Multi-Modal Graph Attention that combines Grid-Graph and Region-Graph features
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        grid_embed_dim: int = 96,
        grid_hidden_dim: int = 128,
        num_regions: int = 100,
        region_feature_dim: int = 512,
        num_attention_heads: int = 8,
        fusion_dim: int = 512,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_regions = num_regions
        self.fusion_dim = fusion_dim

        # Import the previously defined modules
        from grid_graph_feature import GridGraphFeature
        from region_graph_feature import RegionGraphFeature

        # Grid-Graph Feature Extractor
        self.grid_graph_extractor = GridGraphFeature(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=grid_embed_dim,
            gcn_hidden_dim=grid_hidden_dim,
        )

        # Region-Graph Feature Extractor
        self.region_graph_extractor = RegionGraphFeature(
            num_regions=num_regions,
            region_feature_dim=region_feature_dim,
            num_attention_heads=num_attention_heads,
            output_dim=region_feature_dim,
        )

        # Region-Grid Aggregator
        self.region_grid_aggregator = RegionGridAggregator(
            grid_feature_dim=grid_hidden_dim,
            region_feature_dim=region_feature_dim,
            hidden_dim=fusion_dim,
            num_regions=num_regions,
        )

        # Feature fusion layers
        self.region_fusion = nn.Sequential(
            nn.Linear(region_feature_dim + grid_hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self.grid_fusion = nn.Sequential(
            nn.Linear(grid_hidden_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(
        self, images: torch.Tensor, region_boxes: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Multi-Modal Graph Attention
        Args:
            images: [B, 3, H, W] input images
            region_boxes: [B, P, 4] region boxes (optional)
        Returns:
            Dictionary containing:
                - 'fused_features': [B, P+M*k, fusion_dim] fused multi-modal features
                - 'grid_features': [B, M*k, fusion_dim] processed grid features
                - 'region_features': [B, P, fusion_dim] enhanced region features
                - 'region_aggregation': [B, P, grid_hidden_dim] region-grid aggregation
        """
        B = images.shape[0]
        device = images.device

        # Disable mixed precision to avoid dtype issues
        with torch.cuda.amp.autocast(enabled=False):
            # Ensure input is float32
            images = images.float()
            if region_boxes is not None:
                region_boxes = region_boxes.float()

            # Step 1: Extract Grid-Graph Features
            grid_features = self.grid_graph_extractor(
                images
            )  # [B, M*k, grid_hidden_dim]

            # Step 2: Extract Region-Graph Features
            region_features = self.region_graph_extractor(
                images, region_boxes
            )  # [B, P, region_feature_dim]

            # Step 3: Apply Region-Grid Aggregator
            region_aggregation = self.region_grid_aggregator(
                grid_features, region_features
            )  # [B, P, grid_hidden_dim]

            # Step 4: Fuse region features with aggregated grid features
            concatenated_region = torch.cat(
                [region_features, region_aggregation], dim=-1
            )  # [B, P, region_dim + grid_dim]
            enhanced_region_features = self.region_fusion(
                concatenated_region
            )  # [B, P, fusion_dim]

            # Step 5: Process grid features
            processed_grid_features = self.grid_fusion(
                grid_features
            )  # [B, M*k, fusion_dim]

            # Step 6: Apply cross-modal attention between regions and grids
            # Region features attend to grid features
            region_attended, _ = self.cross_attention(
                enhanced_region_features,  # Query: regions
                processed_grid_features,  # Key: grids
                processed_grid_features,  # Value: grids
            )  # [B, P, fusion_dim]

            # Grid features attend to region features
            grid_attended, _ = self.cross_attention(
                processed_grid_features,  # Query: grids
                enhanced_region_features,  # Key: regions
                enhanced_region_features,  # Value: regions
            )  # [B, M*k, fusion_dim]

            # Step 7: Combine attended features
            final_region_features = torch.cat(
                [enhanced_region_features, region_attended], dim=-1
            )  # [B, P, 2*fusion_dim]
            final_grid_features = torch.cat(
                [processed_grid_features, grid_attended], dim=-1
            )  # [B, M*k, 2*fusion_dim]

            # Step 8: Final projection
            final_region_features = self.output_projection(
                final_region_features
            )  # [B, P, fusion_dim]
            final_grid_features = self.output_projection(
                final_grid_features
            )  # [B, M*k, fusion_dim]

            # Step 9: Concatenate all features for final representation
            fused_features = torch.cat(
                [final_region_features, final_grid_features], dim=1
            )  # [B, P+M*k, fusion_dim]

        return {
            "fused_features": fused_features,
            "grid_features": final_grid_features,
            "region_features": final_region_features,
            "region_aggregation": region_aggregation,
        }


# Example usage and testing
def test_mmgat():
    """Test the Multi-Modal Graph Attention implementation"""

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = MultiModalGraphAttention(
        image_size=224,
        patch_size=4,
        grid_embed_dim=96,
        grid_hidden_dim=128,
        num_regions=50,
        region_feature_dim=512,
        num_attention_heads=8,
        fusion_dim=512,
    ).to(device)

    # Create dummy input
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)

    print(f"Input images shape: {dummy_images.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images)

    # Print output shapes
    print("\n--- Output Shapes ---")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

    # Calculate expected dimensions
    num_patches = (224 // 4) ** 2  # 56 * 56 = 3136
    num_regions = 50
    total_features = num_regions + num_patches

    print(f"\n--- Expected Dimensions ---")
    print(f"Number of patches: {num_patches}")
    print(f"Number of regions: {num_regions}")
    print(f"Total features: {total_features}")
    print(f"Fusion dimension: 512")

    # Test with custom region boxes
    print("\n--- Testing with custom region boxes ---")
    custom_boxes = torch.tensor(
        [
            [[10, 10, 50, 50], [60, 60, 100, 100], [120, 30, 180, 90]],
            [[20, 20, 80, 80], [90, 10, 150, 70], [30, 100, 90, 160]],
        ],
        dtype=torch.float32,
    ).to(device)

    # Note: This would require adjusting num_regions to 3 for this test
    # For demonstration, we'll skip this test
    print("Custom box testing skipped (requires model reconfiguration)")

    print("\n--- MMGAT Test Completed Successfully ---")


if __name__ == "__main__":
    test_mmgat()
