"""
Open Vocabulary Query Tool for CLIP features.

Given a text query, find the most similar instances from pre-computed CLIP features.

Usage:
    python tools/open_vocabulary_query.py \
        --query "a red chair" \
        --features_path ./lab9_data/dbscan_masks/clip_features.npy \
        --top_k 5

    # Interactive mode
    python tools/open_vocabulary_query.py \
        --features_path ./lab9_data/dbscan_masks/clip_features.npy \
        --interactive
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed, install it with `pip install open-clip-torch`")


class OpenCLIPTextEncoder:
    """Encode text queries using OpenCLIP."""

    def __init__(self, model_type="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_type,
            pretrained=pretrained,
            precision="fp16",
        )
        self.model.eval()
        self.model.to(device)

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_type)

    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode text queries.

        Args:
            texts: str or list of str
        Returns:
            features: (N, 512) normalized features as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        tokens = self.tokenizer(texts).to(self.device)

        # Encode
        features = self.model.encode_text(tokens)

        # L2 normalize
        features = F.normalize(features, dim=-1)

        return features.cpu().float().numpy()


def load_clip_features(features_path):
    """
    Load pre-computed CLIP features.

    Args:
        features_path: path to .npy file containing features dict
    Returns:
        labels: list of instance labels (ints)
        features: (N, 512) numpy array of features
    """
    data = np.load(features_path, allow_pickle=True).item()

    labels = list(data.keys())
    features = np.stack([data[label] for label in labels], axis=0)  # (N, 512)
    features = features.astype(np.float32)

    # Ensure normalized
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    return labels, features


def query_similar(text_features, instance_features, instance_labels, top_k=5):
    """
    Find most similar instances to text query.

    Args:
        text_features: (1, 512) or (512,) text feature
        instance_features: (N, 512) instance features
        instance_labels: list of N labels
        top_k: number of results to return
    Returns:
        results: list of (label, similarity_score) tuples
    """
    if text_features.ndim == 1:
        text_features = text_features[np.newaxis, :]

    # Compute cosine similarity
    similarities = np.dot(instance_features, text_features.T).squeeze()  # (N,)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        label = instance_labels[idx]
        score = similarities[idx]
        results.append((label, float(score)))

    return results


def main(args):
    print("Loading CLIP features...")
    labels, features = load_clip_features(args.features_path)
    print(f"Loaded {len(labels)} instance features")

    print("\nLoading OpenCLIP model for text encoding...")
    encoder = OpenCLIPTextEncoder(device=args.device)
    print("Model loaded!")

    if args.interactive:
        # Interactive mode
        print("\n" + "="*50)
        print("Open Vocabulary Query - Interactive Mode")
        print("Type your query and press Enter. Type 'quit' to exit.")
        print("="*50 + "\n")

        while True:
            query = input("Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not query:
                continue

            # Encode query
            text_features = encoder.encode_text(query)

            # Find similar instances
            results = query_similar(text_features, features, labels, top_k=args.top_k)

            print(f"\nTop {args.top_k} matches for '{query}':")
            print("-" * 40)
            for rank, (label, score) in enumerate(results, 1):
                print(f"  {rank}. Instance {label:4d}  |  Similarity: {score:.4f}")
            print()

    else:
        # Single query mode
        if not args.query:
            print("Error: --query is required in non-interactive mode")
            return

        # Encode query
        print(f"\nQuery: '{args.query}'")
        text_features = encoder.encode_text(args.query)

        # Find similar instances
        results = query_similar(text_features, features, labels, top_k=args.top_k)

        print(f"\nTop {args.top_k} matches:")
        print("-" * 40)
        for rank, (label, score) in enumerate(results, 1):
            print(f"  {rank}. Instance {label:4d}  |  Similarity: {score:.4f}")

        # Return the best match label for programmatic use
        if results:
            best_label, best_score = results[0]
            print(f"\nBest match: Instance {best_label} (similarity: {best_score:.4f})")
            return best_label


def query_from_code(features_path, query, device="cuda", top_k=5):
    """
    Programmatic interface for open vocabulary query.

    Args:
        features_path: path to clip_features.npy
        query: text query string
        device: cuda or cpu
        top_k: number of results
    Returns:
        results: list of (label, similarity_score) tuples
    """
    labels, features = load_clip_features(features_path)
    encoder = OpenCLIPTextEncoder(device=device)
    text_features = encoder.encode_text(query)
    results = query_similar(text_features, features, labels, top_k=top_k)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Vocabulary Query for CLIP features')
    parser.add_argument('--query', type=str, default=None, help='Text query (e.g., "a red chair")')
    parser.add_argument('--features_path', type=str, required=True, help='Path to clip_features.npy')
    parser.add_argument('--top_k', type=int, default=2, help='Number of top results to return')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')

    args = parser.parse_args()
    main(args)
