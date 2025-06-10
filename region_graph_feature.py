import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict
import torchvision.ops as ops


class FasterRCNN(nn.Module):
    """Simplified Faster R-CNN for object region detection"""

    def __init__(self, backbone_dim: int = 512, num_classes: int = 80):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes

        # Simplified backbone (normally ResNet)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, backbone_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ROI pooling output dimension
        self.roi_pool_size = 7
        self.roi_pooling = ops.RoIPool(
            output_size=(self.roi_pool_size, self.roi_pool_size), spatial_scale=1 / 16
        )

        # Feature extraction after ROI pooling
        self.roi_head = nn.Sequential(
            nn.Linear(backbone_dim * self.roi_pool_size * self.roi_pool_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract features for given region boxes
        Args:
            images: [B, 3, H, W] input images
            boxes: [B, P, 4] region boxes in format [x1, y1, x2, y2]
        Returns:
            region_features: [B, P, 512] features for each region
        """
        B, _, H, W = images.shape
        P = boxes.shape[1]

        # Extract backbone features
        backbone_features = self.backbone(images)  # [B, backbone_dim, H', W']

        # Ensure boxes are on the same device as images
        boxes = boxes.to(images.device)

        # Prepare boxes for ROI pooling
        batch_indices = (
            torch.arange(B, device=images.device).view(B, 1).expand(B, P).reshape(-1)
        )  # [B*P]
        flat_boxes = boxes.view(-1, 4)  # [B*P, 4]
        roi_boxes = torch.cat(
            [batch_indices[:, None], flat_boxes], dim=1
        )  # [B*P, 5] [batch_idx, x1, y1, x2, y2]

        # ROI pooling
        pooled_features = self.roi_pooling(
            backbone_features, roi_boxes
        )  # [B*P, backbone_dim, 7, 7]

        # Flatten and process through ROI head
        pooled_flat = pooled_features.view(B * P, -1)  # [B*P, backbone_dim * 7 * 7]
        roi_features = self.roi_head(pooled_flat)  # [B*P, 512]

        # Reshape back to [B, P, 512]
        region_features = roi_features.view(B, P, 512)

        return region_features


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for region features"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [B, P, d_model] region features
        Returns:
            output: [B, P, d_model] attended features
        """
        B, P, d_model = query.shape

        # Linear projections
        Q = self.w_q(query)  # [B, P, d_model]
        K = self.w_k(key)  # [B, P, d_model]
        V = self.w_v(value)  # [B, P, d_model]

        # Reshape for multi-head attention
        Q = Q.view(B, P, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [B, num_heads, P, d_k]
        K = K.view(B, P, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [B, num_heads, P, d_k]
        V = V.view(B, P, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [B, num_heads, P, d_k]

        # Scaled dot-product attention
        attention_scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # [B, num_heads, P, P]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # [B, num_heads, P, d_k]

        # Concatenate heads
        attended_values = (
            attended_values.transpose(1, 2).contiguous().view(B, P, d_model)
        )

        # Final linear projection
        output = self.w_o(attended_values)

        return output


class SelfAttention(nn.Module):
    """Self-Attention mechanism as described in the paper"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Compute QK^T Self-Attention
        Args:
            R: [B, P, d] region features where Q = K = V = R
        Returns:
            attention_output: [B, P, d] attended features
        """
        # Q = K = V = R
        Q = K = V = R  # [B, P, d]

        # Compute attention scores: QK^T / sqrt(d)
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # [B, P, P]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, P, P]
        attention_output = torch.bmm(attention_weights, V)  # [B, P, d]

        return attention_output


class RegionGraphFeature(nn.Module):
    """
    Region Graph Feature extraction using Faster R-CNN and Multi-Head Attention
    """

    def __init__(
        self,
        num_regions: int = 100,
        region_feature_dim: int = 512,
        num_attention_heads: int = 8,
        output_dim: int = 512,
    ):
        super().__init__()

        self.num_regions = num_regions
        self.region_feature_dim = region_feature_dim
        self.num_attention_heads = num_attention_heads
        self.output_dim = output_dim

        # Faster R-CNN for region detection and feature extraction
        self.faster_rcnn = FasterRCNN(backbone_dim=512, num_classes=80)

        # Self-Attention mechanism
        self.self_attention = SelfAttention(region_feature_dim)

        # Multi-Head Attention mechanism
        self.multi_head_attention = MultiHeadAttention(
            d_model=region_feature_dim, num_heads=num_attention_heads
        )

        # Final projection layer
        self.output_projection = nn.Linear(region_feature_dim, output_dim)

    def generate_dummy_boxes(
        self,
        batch_size: int,
        num_boxes: int,
        image_size: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate dummy bounding boxes for testing
        In practice, these would come from Faster R-CNN's RPN
        Args:
            batch_size: Number of images in batch
            num_boxes: Number of boxes per image
            image_size: (H, W) of input images
            device: Device to place tensors on (CPU or CUDA)
        """
        H, W = image_size

        # Vectorized box generation
        x1 = torch.randint(0, W // 2, (batch_size, num_boxes), device=device).float()
        y1 = torch.randint(0, H // 2, (batch_size, num_boxes), device=device).float()
        x2 = (
            x1
            + torch.randint(
                W // 4, W // 2, (batch_size, num_boxes), device=device
            ).float()
        )
        y2 = (
            y1
            + torch.randint(
                H // 4, H // 2, (batch_size, num_boxes), device=device
            ).float()
        )

        # Clamp boxes to image bounds
        x2 = torch.clamp(x2, max=W - 1)
        y2 = torch.clamp(y2, max=H - 1)

        # Stack coordinates: [x1, y1, x2, y2]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [batch_size, num_boxes, 4]

        return boxes

    def forward(
        self, images: torch.Tensor, region_boxes: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for region graph feature extraction
        Args:
            images: [B, 3, H, W] input images
            region_boxes: [B, P, 4] region boxes (optional, will generate dummy if None)
        Returns:
            R^(0): [B, P, output_dim] initial region representation
        """
        B, C, H, W = images.shape
        device = images.device  # Get device from input images

        # Generate dummy boxes if not provided
        if region_boxes is None:
            region_boxes = self.generate_dummy_boxes(
                B, self.num_regions, (H, W), device
            )
        else:
            region_boxes = region_boxes.to(device)

        # Enable mixed precision for faster computation on GPU
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            # Step 1: Extract region features using Faster R-CNN
            region_features = self.faster_rcnn(images, region_boxes)  # R: [B, P, d]

            # Step 2: Apply Self-Attention mechanism
            sa_output = self.self_attention(region_features)  # [B, P, d]

            # Step 3: Apply Multi-Head Attention
            mha_output = self.multi_head_attention(
                sa_output, sa_output, sa_output
            )  # [B, P, d]

            # Step 4: Combine with residual connection
            combined_features = region_features + mha_output  # [B, P, d]

            # Step 5: Final projection to get R^(0)
            initial_region_representation = self.output_projection(
                combined_features
            )  # [B, P, output_dim]

        return initial_region_representation

    def get_region_vectors(
        self, region_representation: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Extract individual region vectors {r_i}
        Args:
            region_representation: [B, P, d] region features
        Returns:
            List of region vectors for each batch
        """
        B, P, d = region_representation.shape
        region_vectors = []

        for b in range(B):
            batch_vectors = [region_representation[b, p, :] for p in range(P)]
            region_vectors.append(batch_vectors)

        return region_vectors


# Example usage and testing
def test_region_graph_feature():
    """Test the Region Graph Feature implementation"""

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model and move to device
    model = RegionGraphFeature(
        num_regions=50, region_feature_dim=512, num_attention_heads=8, output_dim=512
    ).to(device)

    # Create dummy input and move to device
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)

    print(f"Input images shape: {dummy_images.shape}")
    print(f"Number of regions: {model.num_regions}")
    print(f"Region feature dimension: {model.region_feature_dim}")

    # Forward pass
    with torch.no_grad():
        region_representation = model(dummy_images)

    print(f"Output region representation shape: {region_representation.shape}")
    print(f"Expected shape: [{batch_size}, {model.num_regions}, {model.output_dim}]")

    # Test with custom boxes
    print("\n--- Testing with custom region boxes ---")
    custom_boxes = torch.tensor(
        [
            [[10, 10, 50, 50], [60, 60, 100, 100], [120, 30, 180, 90]],
            [[20, 20, 80, 80], [90, 10, 150, 70], [30, 100, 90, 160]],
        ],
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        custom_region_representation = model(dummy_images, custom_boxes)

    print(f"Custom region representation shape: {custom_region_representation.shape}")

    # Test individual components
    print("\n--- Testing individual components ---")

    # Test Faster R-CNN feature extraction
    dummy_boxes = model.generate_dummy_boxes(batch_size, 10, (224, 224), device)
    region_features = model.faster_rcnn(dummy_images, dummy_boxes)
    print(f"Faster R-CNN features shape: {region_features.shape}")

    # Test Self-Attention
    sa_output = model.self_attention(region_features)
    print(f"Self-Attention output shape: {sa_output.shape}")

    # Test Multi-Head Attention
    mha_output = model.multi_head_attention(
        region_features, region_features, region_features
    )
    print(f"Multi-Head Attention output shape: {mha_output.shape}")

    # Extract individual region vectors
    region_vectors = model.get_region_vectors(region_representation)
    print(f"Number of region vector batches: {len(region_vectors)}")
    print(f"Number of vectors per batch: {len(region_vectors[0])}")
    print(f"Each region vector shape: {region_vectors[0][0].shape}")


if __name__ == "__main__":
    test_region_graph_feature()
