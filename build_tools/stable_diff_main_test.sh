rm -rf ./test_images
mkdir test_images
python shark/examples/shark_inference/stable_diffusion/main.py --device=vulkan --output_dir=./test_images --no-load_vmfb --no-use_tuned
python shark/examples/shark_inference/stable_diffusion/main.py --device=vulkan --output_dir=./test_images --no-load_vmfb --no-use_tuned --beta_models=True

python build_tools/image_comparison.py -n ./test_images/*.png
exit $?
