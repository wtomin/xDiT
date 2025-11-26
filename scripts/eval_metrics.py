import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from cleanfid import fid as clean_fid_module

def calculate_clip_score(image_dir, prompt_file):
    """
    Calculates CLIP Score for images in a directory against prompts in a file.
    Assumes images are named sequentially or mapped to lines in prompt_file.
    """
    print(f"📊 Calculating CLIP Score for {image_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load prompts
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    if len(image_paths) != len(prompts):
        print(f"[Warning] Number of images ({len(image_paths)}) does not match number of prompts ({len(prompts)}). CLIP score might be inaccurate.")
        # Truncate to the shorter length for safety
        min_len = min(len(image_paths), len(prompts))
        image_paths = image_paths[:min_len]
        prompts = prompts[:min_len]

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
    
    scores = []
    for img_path, prompt in zip(image_paths, prompts):
        image = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
        
        # Compute score for single pair
        score = metric(img_tensor.unsqueeze(0), [prompt])
        scores.append(score.item())
    
    avg_score = sum(scores) / len(scores)
    print(f"✅ Average CLIP Score: {avg_score:.4f}")
    return avg_score

def calculate_fid_score(pred_dir, gt_dir):
    """
    Calculates FID Score between a prediction directory and a ground truth directory.
    """
    print(f"📊 Calculating FID Score between {pred_dir} (Pred) and {gt_dir} (GT)...")
    if clean_fid_module is None:
                raise ImportError("Clean-FID is not installed. Please run `pip install clean-fid`.")
    print("🔹 Using Clean-FID (Parmar et al., 2022) protocol...")
    # Clean-FID computes statistics internally.
    # Note: The paper says images are resized to 256px. 
    # If input images are already 256px, clean-fid uses them as is if no resizing mode is forced,
    # but usually it resizes to 299 for Inception. 
    # However, since we pre-resized images to 256px, we pass them directly.
    score = clean_fid_module.compute_fid(gt_dir, pred_dir)
    print(f"✅ Clean-FID Score: {score:.4f}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing generated images (Prediction)")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing ground truth images (for FID)")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to text file containing prompts (for CLIP)")
    parser.add_argument("--metrics", nargs="+", default=["fid"], choices=["clip", "fid"], help="Metrics to calculate")
    
    args = parser.parse_args()

    if "clip" in args.metrics:
        if args.prompt_file:
            calculate_clip_score(args.pred_dir, args.prompt_file)
        else:
            print("[Skipping CLIP] --prompt_file is required.")

    if "fid" in args.metrics:
        if args.gt_dir:
            calculate_fid_score(args.pred_dir, args.gt_dir)
        else:
            print("[Skipping FID] --gt_dir is required.")