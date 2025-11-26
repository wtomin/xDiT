import os
import json
import random
import argparse
import requests
import zipfile
from PIL import Image
from tqdm import tqdm
from io import BytesIO

def download_and_extract(url, extract_to):
    """Downloads and extracts a zip file if not already present."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def prepare_coco_dataset(output_dir, num_samples=10000, seed=42, download=False):
    """
    Prepares the COCO validation set for FID evaluation.
    1. Downloads val2014 images and annotations.
    2. Samples 30k images/captions.
    3. Resizes GT images to 256x256.
    4. Saves captions to a JSON file for the generator.
    """
    random.seed(seed)
    
    # Paths
    data_root = os.path.join(output_dir, "coco_raw")
    gt_resized_dir = os.path.join(output_dir, "coco_256_gt_"+str(num_samples))
    os.makedirs(gt_resized_dir, exist_ok=True)
    if download:
        print("Starting COCO dataset download and preparation...")
        # URLs for COCO 2014 Val
        img_url = "http://images.cocodataset.org/zips/val2014.zip"
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        # Download
        download_and_extract(img_url, data_root)
        download_and_extract(ann_url, data_root)
    
    # Load Annotations
    ann_file = os.path.join(data_root, "annotations", "captions_val2014.json")
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    # Map image_id to filename
    img_map = {img['id']: img['file_name'] for img in coco['images']}
    # Map image_id to captions (take the first one)
    img_to_caption = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_caption:
            img_to_caption[img_id] = ann['caption']
            
    # Filter images that have both file and caption
    valid_img_ids = [iid for iid in img_map.keys() if iid in img_to_caption]
    
    # Sample num_samples
    if len(valid_img_ids) < num_samples:
        print(f"[Warning] Only {len(valid_img_ids)} valid images found. Using all.")
        selected_ids = valid_img_ids
    else:
        selected_ids = random.sample(valid_img_ids, num_samples)
        
    print(f"Processing {len(selected_ids)} images...")
    
    prompts_data = []
    
    for i, img_id in tqdm(enumerate(selected_ids), total=len(selected_ids)):
        filename = img_map[img_id]
        caption = img_to_caption[img_id]
        
        src_path = os.path.join(data_root, "val2014", filename)
        dst_filename = f"{i:05d}.png" # Rename to index for alignment
        dst_path = os.path.join(gt_resized_dir, dst_filename)
        
        # Resize GT to 256x256 as per paper
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img_resized = img.resize((256, 256), Image.BICUBIC)
                img_resized.save(dst_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
        prompts_data.append({
            "id": i,
            "original_filename": filename,
            "prompt": caption
        })
        
    # Save prompts for inference
    prompt_file = os.path.join(output_dir, "coco_"+str(num_samples)+"_prompts.json")
    with open(prompt_file, 'w') as f:
        json.dump(prompts_data, f, indent=2)
        
    print(f"✅ Preparation complete.")
    print(f"GT Images (256px): {gt_resized_dir}")
    print(f"Prompts File: {prompt_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/scratch/fq9hpsac/coco_benchmark", help="Root directory for data")
    args = parser.parse_args()
    prepare_coco_dataset(args.output_dir)