#!/bin/bash

# ==============================================================================
# xDiT COCO Benchmark (Method 2: PipeFusion Paper Reproduction)
# 1. Prepare COCO Validation Set (Resize to 256px)
# 2. Generate 256px Images using xDiT (Multi-GPU)
# 3. Calculate FID
# ==============================================================================

set -e # Exit on error

# --- Configuration ---
# Paths
MODEL_PATH=${1:-"/scratch/fq9hpsac/black-forest-labs/FLUX.1-dev"} # Default model
DATA_ROOT="/scratch/fq9hpsac/coco_benchmark"
NUM_SAMPLES=10000 # As requested
SCRIPT="flux_example_multi.py"


# Parallel Settings (Adjust based on your hardware, e.g., 8 GPUs)
NUM_GPUS=1
PIPEFUSION_DEGREE=1
ULYSSES_DEGREE=1 # 2*2 = 4 GPUs per instance (allows DP=2 on 8 GPUs) or adjust as needed
# Ensure N_GPUS matches the degrees. 
# Example: If 8 GPUs, PP=2, Ulysses=2 -> Total Parallel=4 -> Data Parallel=2

# Output Directories
GT_DIR="${DATA_ROOT}/coco_256_gt_${NUM_SAMPLES}"
PROMPT_FILE="${DATA_ROOT}/coco_${NUM_SAMPLES}_prompts.json"
PRED_DIR="${DATA_ROOT}/results_pipefusion_${NUM_SAMPLES}"
export PYTHONPATH=$PWD:$PYTHONPATH

echo "========================================================"
echo "Starting COCO FID Benchmark"
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Resolution: 256x256"
echo "========================================================"

# Note: We force height/width to 256 to match the paper's benchmark setting
# We use output_type pil to ensure images are saved for FID calculation

torchrun --nproc_per_node=$NUM_GPUS ./examples/$SCRIPT \
    --model $MODEL_PATH \
    --output_dir "$PRED_DIR" \
    --prompt_file "$PROMPT_FILE" \
    --height 256 \
    --width 256 \
    --pipefusion_parallel_degree $PIPEFUSION_DEGREE \
    --ulysses_degree $ULYSSES_DEGREE \
    --num_inference_steps 28 \
    --warmup_steps 1 \
    --use_torch_compile \
    --seed 42

echo "✅ Generation Complete. Images saved to $PRED_DIR"

# --- Step 3: Evaluate FID ---
echo "--------------------------------------------------------"
echo "📊 Step 3: Calculating FID Score..."
echo "--------------------------------------------------------"

# Using the eval_metrics.py we defined previously
# It compares the generated folder against the resized GT folder
echo "PRED_DIR: $PRED_DIR"
echo "GT_DIR: $GT_DIR"
python scripts/eval_metrics.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --metrics fid

echo "========================================================"
echo "Benchmark Finished."