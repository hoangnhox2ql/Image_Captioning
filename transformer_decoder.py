import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        Q = (
            self.W_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (
            self.W_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        return self.W_o(attention_output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Self-attention block
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention block
        attn_output = self.cross_attention(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class MMGATImageCaptioning(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=50,
        feature_dim=2048,
    ):
        super(MMGATImageCaptioning, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Feature projection layers
        self.feature_projection = nn.Linear(feature_dim, d_model)

        # Word embedding and positional encoding
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(
            max_seq_length, d_model
        )

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(0.1)

    def create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def create_causal_mask(self, seq_length):
        """Create causal mask for self-attention"""
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, combined_features, target_sequence=None, max_length=20):
        """
        Args:
            combined_features: Combined features from MMGAT [batch_size, num_regions, feature_dim]
            target_sequence: Target sequence for training [batch_size, seq_length]
            max_length: Maximum generation length for inference
        """
        batch_size = combined_features.size(0)

        # Project image features to model dimension
        memory = self.feature_projection(
            combined_features
        )  # [batch_size, num_regions, d_model]

        if target_sequence is not None:
            # Training mode
            return self.forward_train(memory, target_sequence)
        else:
            # Inference mode
            return self.forward_inference(memory, max_length)

    def forward_train(self, memory, target_sequence):
        batch_size, seq_length = target_sequence.size()

        # Word embeddings + positional encoding
        word_embeds = self.word_embedding(target_sequence) * math.sqrt(self.d_model)
        pos_encoding = self.positional_encoding[:, :seq_length, :].to(
            target_sequence.device
        )
        decoder_input = self.dropout(word_embeds + pos_encoding)

        # Create causal mask
        causal_mask = self.create_causal_mask(seq_length).to(target_sequence.device)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input, memory, tgt_mask=causal_mask)

        # Output projection
        logits = self.output_projection(decoder_input)

        return logits

    def forward_inference(self, memory, max_length):
        batch_size = memory.size(0)
        device = memory.device

        # Start with <BOS> token (assuming index 1)
        generated_sequence = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length):
            # Word embeddings + positional encoding
            word_embeds = self.word_embedding(generated_sequence) * math.sqrt(
                self.d_model
            )
            seq_length = generated_sequence.size(1)
            pos_encoding = self.positional_encoding[:, :seq_length, :].to(device)
            decoder_input = self.dropout(word_embeds + pos_encoding)

            # Create causal mask
            causal_mask = self.create_causal_mask(seq_length).to(device)

            # Pass through decoder layers
            for layer in self.decoder_layers:
                decoder_input = layer(decoder_input, memory, tgt_mask=causal_mask)

            # Get prediction for next token
            logits = self.output_projection(decoder_input[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)

            # Append to sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            # Check for <EOS> token (assuming index 2)
            if (next_token == 2).all():
                break

        return generated_sequence


# Example usage
def example_usage():
    # Model parameters
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    feature_dim = 2048  # Combined feature dimension from MMGAT

    # Initialize model
    model = MMGATImageCaptioning(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        feature_dim=feature_dim,
    )

    # Example input - combined features from MMGAT
    batch_size = 2
    num_regions = 49  # e.g., 7x7 grid regions
    combined_features = torch.randn(batch_size, num_regions, feature_dim)

    # Training example
    target_sequence = torch.randint(0, vocab_size, (batch_size, 20))
    train_logits = model(combined_features, target_sequence)
    print(f"Training logits shape: {train_logits.shape}")

    # Inference example
    model.eval()
    with torch.no_grad():
        generated_captions = model(combined_features, max_length=20)
        print(f"Generated captions shape: {generated_captions.shape}")

    return model


# Training function
def train_step(model, combined_features, target_sequence, criterion, optimizer):
    model.train()

    # Forward pass
    logits = model(combined_features, target_sequence[:, :-1])  # Exclude last token

    # Calculate loss
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        target_sequence[:, 1:].reshape(-1),  # Exclude first token
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    model = example_usage()
