import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import spacy
import networkx as nx


class BERTEmbedding(nn.Module):
    """BERT-based text embedding for semantic features"""

    def __init__(self, model_name: str = "bert-base-uncased", embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)

        # Freeze BERT parameters for efficiency (optional)
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, text_sequences: List[str]) -> torch.Tensor:
        """
        Convert text sequences to BERT embeddings
        Args:
            text_sequences: List of text strings
        Returns:
            embeddings: [N, embedding_dim] where N is number of sequences
        """
        # Tokenize texts
        encoded = self.tokenizer(
            text_sequences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**encoded)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # [N, embedding_dim]

        return embeddings


class DependencyParser:
    """Dependency parsing for constructing semantic graphs"""

    def __init__(self):
        # Load spaCy model for dependency parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Warning: spaCy English model not found. Please run 'python -m spacy download en_core_web_sm' to install."
            )
            self.nlp = None

    def parse_text(self, text: str) -> List[Dict]:
        """
        Parse text and extract dependency relationships
        Args:
            text: Input text string
        Returns:
            dependencies: List of dependency relations
        """
        if self.nlp is None:
            # Dummy dependencies for testing
            words = text.split()
            return [
                {"head": i, "child": i + 1, "relation": "dummy"}
                for i in range(len(words) - 1)
            ]

        doc = self.nlp(text)
        dependencies = []

        for token in doc:
            if token.head != token:  # Skip root
                dependencies.append(
                    {
                        "head": token.head.i,
                        "child": token.i,
                        "relation": token.dep_,
                        "head_text": token.head.text,
                        "child_text": token.text,
                    }
                )

        return dependencies

    def extract_words(self, text: str) -> List[str]:
        """Extract individual words from text"""
        if self.nlp is None:
            return text.split()

        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_punct]


class SemanticGraphConstructor:
    """Construct semantic graphs from dependency trees"""

    def __init__(self):
        self.dependency_parser = DependencyParser()

    def construct_adjacency_matrix(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Construct adjacency matrix from dependency tree
        Args:
            text: Input text string
        Returns:
            adjacency_matrix: [N, N] adjacency matrix
            words: List of words corresponding to matrix indices
        """
        # Extract words and dependencies
        words = self.dependency_parser.extract_words(text)
        dependencies = self.dependency_parser.parse_text(text)

        N = len(words)
        if N == 0:
            return torch.zeros(1, 1), [""]

        # Initialize adjacency matrix
        adjacency_matrix = torch.zeros(N, N)

        # Fill adjacency matrix based on dependencies
        for dep in dependencies:
            head_idx = dep["head"]
            child_idx = dep["child"]

            # Ensure indices are within bounds
            if 0 <= head_idx < N and 0 <= child_idx < N:
                # Undirected graph: set both directions
                adjacency_matrix[head_idx, child_idx] = 1.0
                adjacency_matrix[child_idx, head_idx] = 1.0

        # Add self-connections (diagonal)
        for i in range(N):
            adjacency_matrix[i, i] = 1.0

        return adjacency_matrix, words

    def correlation_function(self, wi: str, wj: str) -> float:
        """
        Compute correlation between two words D(wi, wj)
        This is a simplified implementation - in practice, could use:
        - Word embeddings similarity
        - Co-occurrence statistics
        - Semantic similarity measures
        """
        if wi == wj:
            return 1.0

        # Jaccard similarity of character sets
        set_i = set(wi.lower())
        set_j = set(wj.lower())

        if len(set_i.union(set_j)) == 0:
            return 0.0

        correlation = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
        return correlation


class SemanticGraphConvolutionalLayer(nn.Module):
    """
    Semantic Graph Convolutional Layer
    S^(0) = σ(Ã^S σ(Ã^S SW₁^S) W₂^S)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weight matrices
        self.W1 = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1.0 / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1.0 / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv2, stdv2)

    def normalize_adjacency_matrix(
        self, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize adjacency matrix: Ã = D^(-1/2) A D^(-1/2)
        """
        # Compute degree matrix
        degree = torch.sum(adjacency_matrix, dim=-1, keepdim=True)  # [N, 1]
        degree = torch.clamp(degree, min=1.0)  # Avoid division by zero

        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)

        # Normalize: D^(-1/2) A D^(-1/2)
        normalized_adj = (
            degree_inv_sqrt * adjacency_matrix * degree_inv_sqrt.transpose(-2, -1)
        )

        return normalized_adj

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, input_dim] node features S
            adjacency_matrix: [N, N] adjacency matrix A^S
        Returns:
            output: [N, output_dim] output features S^(0)
        """
        # Normalize adjacency matrix
        normalized_adj = self.normalize_adjacency_matrix(adjacency_matrix)  # Ã^S

        # First GCN layer: Ã^S S W₁^S
        h1 = torch.matmul(node_features, self.W1)  # [N, hidden_dim]
        h1 = torch.matmul(normalized_adj, h1)  # [N, hidden_dim]
        h1 = F.relu(h1)  # σ(Ã^S S W₁^S)

        # Second GCN layer: Ã^S σ(Ã^S S W₁^S) W₂^S
        h2 = torch.matmul(h1, self.W2)  # [N, output_dim]
        output = torch.matmul(normalized_adj, h2)  # [N, output_dim]
        output = F.relu(output)  # σ(Ã^S σ(Ã^S S W₁^S) W₂^S)

        return output


class SemanticGraphFeature(nn.Module):
    """
    Complete Semantic Graph Feature extraction pipeline
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        bert_dim: int = 768,
        gcn_hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()

        self.bert_dim = bert_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.output_dim = output_dim

        # BERT embedding for word-level features
        self.bert_embedding = BERTEmbedding(bert_model_name, bert_dim)

        # Semantic graph constructor
        self.graph_constructor = SemanticGraphConstructor()

        # Semantic Graph Convolutional Layer
        self.semantic_gcn = SemanticGraphConvolutionalLayer(
            input_dim=bert_dim, hidden_dim=gcn_hidden_dim, output_dim=output_dim
        )

    def process_single_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single text to extract semantic graph features
        Args:
            text: Input text string
        Returns:
            semantic_features: [N, output_dim] semantic node features
            adjacency_matrix: [N, N] semantic adjacency matrix
        """
        # Step 1: Construct semantic graph
        adjacency_matrix, words = self.graph_constructor.construct_adjacency_matrix(
            text
        )

        if len(words) == 0 or words == [""]:
            # Handle empty text
            return torch.zeros(1, self.output_dim), torch.zeros(1, 1)

        # Step 2: Get BERT embeddings for words
        word_embeddings = self.bert_embedding(words)  # [N, bert_dim]

        # Step 3: Apply Semantic GCN
        semantic_features = self.semantic_gcn(
            word_embeddings, adjacency_matrix
        )  # [N, output_dim]

        return semantic_features, adjacency_matrix

    def forward(
        self, text_list: List[str]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process multiple texts
        Args:
            text_list: List of text strings
        Returns:
            semantic_features_list: List of [Ni, output_dim] tensors
            adjacency_matrices_list: List of [Ni, Ni] tensors
        """
        semantic_features_list = []
        adjacency_matrices_list = []

        for text in text_list:
            semantic_features, adjacency_matrix = self.process_single_text(text)
            semantic_features_list.append(semantic_features)
            adjacency_matrices_list.append(adjacency_matrix)

        return semantic_features_list, adjacency_matrices_list

    def get_semantic_vectors(
        self, semantic_features_list: List[torch.Tensor]
    ) -> List[List[torch.Tensor]]:
        """
        Extract individual semantic vectors {s_i}
        Args:
            semantic_features_list: List of [Ni, output_dim] tensors
        Returns:
            List of lists containing individual semantic vectors
        """
        semantic_vectors = []

        for semantic_features in semantic_features_list:
            N, d = semantic_features.shape
            text_vectors = [semantic_features[i, :] for i in range(N)]
            semantic_vectors.append(text_vectors)

        return semantic_vectors


# Example usage and testing
def test_semantic_graph_feature():
    """Test the Semantic Graph Feature implementation"""

    print("Testing Semantic Graph Feature Implementation")
    print("=" * 50)

    # Create model
    model = SemanticGraphFeature(
        bert_model_name="bert-base-uncased",
        bert_dim=768,
        gcn_hidden_dim=512,
        output_dim=256,
    )

    # Test texts (image captions)
    test_texts = [
        "A cat sitting on a wooden table",
        "Two dogs playing in the park",
        "Beautiful sunset over the ocean",
        "Person riding bicycle on street",
    ]

    print(f"Test texts: {len(test_texts)} captions")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")

    # Process texts
    semantic_features_list, adjacency_matrices_list = model(test_texts)

    print(f"\nResults:")
    print(f"Number of processed texts: {len(semantic_features_list)}")

    for i, (features, adj_matrix) in enumerate(
        zip(semantic_features_list, adjacency_matrices_list)
    ):
        print(f"\nText {i+1}: '{test_texts[i]}'")
        print(f"  Semantic features shape: {features.shape}")
        print(f"  Adjacency matrix shape: {adj_matrix.shape}")
        print(f"  Number of words/nodes: {features.shape[0]}")
        print(f"  Feature dimension: {features.shape[1]}")
        print(
            f"  Graph density: {(adj_matrix.sum() - adj_matrix.trace()) / (adj_matrix.numel() - adj_matrix.shape[0]):.3f}"
        )

    # Test individual components
    print(f"\n" + "=" * 50)
    print("Testing Individual Components")
    print("=" * 50)

    # Test dependency parsing
    test_text = "A cat sitting on a wooden table"
    dependencies = model.graph_constructor.dependency_parser.parse_text(test_text)
    words = model.graph_constructor.dependency_parser.extract_words(test_text)

    print(f"\nDependency parsing for: '{test_text}'")
    print(f"Words: {words}")
    print(f"Dependencies: {len(dependencies)}")
    for dep in dependencies[:3]:  # Show first 3
        print(f"  {dep}")

    # Test adjacency matrix construction
    adj_matrix, extracted_words = model.graph_constructor.construct_adjacency_matrix(
        test_text
    )
    print(f"\nAdjacency matrix shape: {adj_matrix.shape}")
    print(f"Extracted words: {extracted_words}")
    print(f"Adjacency matrix:\n{adj_matrix}")

    # Test BERT embeddings
    word_embeddings = model.bert_embedding(words[:3])  # Test first 3 words
    print(f"\nBERT embeddings shape for 3 words: {word_embeddings.shape}")
    print(f"Embedding dimension: {word_embeddings.shape[1]}")

    # Test semantic vectors extraction
    semantic_vectors = model.get_semantic_vectors(semantic_features_list)
    print(f"\nSemantic vectors:")
    print(f"Number of texts: {len(semantic_vectors)}")
    if len(semantic_vectors) > 0:
        print(f"Vectors in first text: {len(semantic_vectors[0])}")
        if len(semantic_vectors[0]) > 0:
            print(f"Each vector shape: {semantic_vectors[0][0].shape}")


if __name__ == "__main__":
    test_semantic_graph_feature()
