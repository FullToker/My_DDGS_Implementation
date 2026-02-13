"""
Extract CLIP features from DBSCAN masks.

For each instance label in the masks:
1. Mask out pixels outside the instance (set to 0)
2. Crop using bounding box
3. Pad to square and resize to 224x224
4. Encode with OpenCLIP
5. Cosine average features across images for the same label

Usage:
    python tools/extract_clip_features_from_masks.py \
        --image_dir ./lab9_data/images \
        --mask_dir ./lab9_data/dbscan_masks
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed, install it with `pip install open-clip-torch`")


class OpenCLIPEncoder:
    def __init__(self, model_type="ViT-B-16", pretrained="laion2b_s34b_b88k", device="cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_type,
            pretrained=pretrained,
            precision="fp16",
        )
        self.model.eval()
        self.model.to(device)

        # Custom preprocessing (matching the reference code)
        self.process = torch.nn.Sequential(
            torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )
        self.normalize = torch.nn.Sequential(
            torch.nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
        )

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, images_tensor):
        """
        Args:
            images_tensor: (N, 3, 224, 224) float tensor in [0, 1]
        Returns:
            features: (N, 512) normalized features
        """
        # Normalize
        images_tensor = (images_tensor - self.mean) / self.std
        images_tensor = images_tensor.half()

        # Encode
        features = self.model.encode_image(images_tensor)

        # L2 normalize
        features = F.normalize(features, dim=-1)

        return features.detach().cpu().half()


def get_seg_img(mask_binary, image):
    """
    Mask out pixels outside the instance and crop using bbox.

    Args:
        mask_binary: (H, W) boolean array, True for the instance
        image: (H, W, 3) uint8 array
    Returns:
        seg_img: cropped image with background set to 0
    """
    image = image.copy()
    image[~mask_binary] = 0

    # Get bounding box
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    seg_img = image[y_min:y_max+1, x_min:x_max+1]
    return seg_img


def pad_img(img):
    """Pad image to square."""
    h, w = img.shape[:2]
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
    return pad


def process_single_instance(mask_binary, image):
    """
    Process a single instance: mask -> crop -> pad -> resize to 224x224.

    Args:
        mask_binary: (H, W) boolean array
        image: (H, W, 3) uint8 array
    Returns:
        tensor: (3, 224, 224) float tensor in [0, 1], or None if invalid
    """
    # Check if mask is valid (has enough pixels)
    if mask_binary.sum() < 10:
        return None

    # Get segmented image
    seg_img = get_seg_img(mask_binary, image)

    # Check if crop is valid
    if seg_img.shape[0] < 1 or seg_img.shape[1] < 1:
        return None

    # Pad to square
    padded = pad_img(seg_img)

    # Resize to 224x224
    from PIL import Image as PILImage
    padded_pil = PILImage.fromarray(padded)
    resized = padded_pil.resize((224, 224), PILImage.BILINEAR)
    resized = np.array(resized)

    # Convert to tensor (C, H, W) in [0, 1]
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor


def cosine_average(features_list):
    """
    Compute cosine average of features.
    Normalize -> Average -> Normalize again.

    Args:
        features_list: list of (512,) tensors
    Returns:
        averaged: (512,) tensor
    """
    if len(features_list) == 1:
        return features_list[0]

    # Stack and ensure normalized
    stacked = torch.stack(features_list, dim=0).float()  # (N, 512)
    stacked = F.normalize(stacked, dim=-1)

    # Average
    averaged = stacked.mean(dim=0)

    # Normalize again
    averaged = F.normalize(averaged, dim=-1)

    return averaged.half()


def main(args):
    # Get file lists
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.endswith('.npy') and 'mask_' in f and 'colored' not in f])

    assert len(image_files) == len(mask_files), \
        f"Number of images ({len(image_files)}) != number of masks ({len(mask_files)})"

    print(f"Found {len(image_files)} images and {len(mask_files)} masks")

    # Load first mask to get target size
    first_mask = np.load(os.path.join(args.mask_dir, mask_files[0]))
    target_h, target_w = first_mask.shape
    print(f"Mask size: {target_w} x {target_h}")

    # Initialize CLIP encoder
    print("Loading OpenCLIP model...")
    encoder = OpenCLIPEncoder(device=args.device)

    # Dictionary to collect features for each label across all images
    label_features = defaultdict(list)

    # Process each image-mask pair
    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing images"):
        # Load image and resize to mask size
        img_path = os.path.join(args.image_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((target_w, target_h), Image.BILINEAR)
        image = np.array(image)

        # Load mask
        mask_path = os.path.join(args.mask_dir, mask_file)
        mask = np.load(mask_path)

        # Get unique labels (excluding background if needed)
        unique_labels = np.unique(mask)

        # Prepare batch for this image
        batch_tensors = []
        batch_labels = []

        for label in unique_labels:
            if args.skip_label is not None and label == args.skip_label:
                continue

            mask_binary = (mask == label)

            # Skip small instances
            if mask_binary.sum() < args.min_pixels:
                continue

            tensor = process_single_instance(mask_binary, image)
            if tensor is not None:
                batch_tensors.append(tensor)
                batch_labels.append(label)

        if len(batch_tensors) == 0:
            continue

        # Batch encode
        batch = torch.stack(batch_tensors, dim=0).to(args.device)  # (N, 3, 224, 224)
        features = encoder.encode(batch)  # (N, 512)

        # Store features
        for label, feat in zip(batch_labels, features):
            label_features[label].append(feat)

    # Cosine average across images
    print("\nComputing cosine average for each label...")
    final_features = {}
    for label, feats in tqdm(label_features.items(), desc="Averaging"):
        final_features[int(label)] = cosine_average(feats).numpy()

    print(f"\nTotal unique labels: {len(final_features)}")

    # Save results
    output_path = os.path.join(args.mask_dir, 'clip_features.npy')
    np.save(output_path, final_features)
    print(f"Saved features to: {output_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Labels processed: {len(final_features)}")
    print(f"Feature dimension: 512")
    print(f"Feature dtype: float16")

    # Show label statistics
    label_counts = {label: len(feats) for label, feats in label_features.items()}
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 labels by occurrence count:")
    for label, count in sorted_counts:
        print(f"  Label {label}: {count} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract CLIP features from DBSCAN masks')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask .npy files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--min_pixels', type=int, default=30, help='Minimum pixels for an instance')
    parser.add_argument('--skip_label', type=int, default=None, help='Label to skip (e.g., background)')

    args = parser.parse_args()
    main(args)
