set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
export MODEL_TYPE="Pixart-sigma"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py PixArt-alpha/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py PixArt-alpha/PixArt-Sigma-XL-2-1024-MS 20"
    ["Sd3"]="sd3_example.py stabilityai/stable-diffusion-3-medium-diffusers 20"
    ["Flux"]="flux_example.py black-forest-labs/FLUX.1-dev 28"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5"

# For single-card, no parallel or device args are needed

# If you want to select a specific visible device, uncomment the appropriate line
# export CUDA_VISIBLE_DEVICES=0
# export ASCEND_RT_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 ./examples_npu/$SCRIPT \
    --model $MODEL_ID \
    $TASK_ARGS \
    --num_inference_steps $INFERENCE_STEP \
    --warmup_steps 1 \
    --prompt "brown dog laying on the ground with a metal bowl in front of him."
