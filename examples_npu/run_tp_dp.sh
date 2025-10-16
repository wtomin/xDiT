set -xe

torchrun --nproc_per_node=4 --start_method=spawn examples_npu/sd3_example.py \
--model /workspace/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers \
--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 3.5 \
--num_inference_steps 50 \
--warmup_steps 1 \
--prompt "brown dog laying on the ground with a metal bowl in front of him." "A small cat." "A good man" \
--tensor_parallel_degree 2 \
--data_parallel_degree 2