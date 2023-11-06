# Stable Diffusion

Description: Export stable diffusion models from huggingface (or checkpoints from CivitAI) to ONNX, then run the model for inference in a NodeJS program.


### Setup

 - It is recommended that you create a python virtual environment instead of a conda due to version issues with a lot of necessary packages.
 - To set up the virtual environment, install the venv package:
	 - `pip3 install virtualenv`
 - Create the new virtual environment:
	 - `python -m venv sd2onnx-env`
 - Activate the virtual environment:
	 - Linux/MacOS: source autogptq-env/bin/activate
	 - Windows: .\sd2onnx-env\Scripts\activate
 - Deactivate the virtual environment:
	 - `deactivate`
 - Install the necessary packages (while the virtual environment is active):
	 - `(sd2onnx-env) pip3 install -r requirements.txt`
 - Also be sure to install the necessary version of pytorch according to your OS (refer to the pytorch website but the following command will help):
 - Linux & Windows (CUDA 11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
 - MacOS: `pip3 install torch torchvision torchaudio`


### Notes

 - Convert stable diffusion model from huggingface hub to onnx via command line:
	 - `optimum-cli export onnx --model model_id/on_hub dest_folder`
	 - Works as expected.
 - Convert stable diffusion model from checkpoint file (`.ckpt` or `.safetensor`) to onnx via `diffusers` scripts:
	 - Clone the `diffusers` git repo:
		 - `git clone https://github.com/huggingface/diffusers.git`
	 - Convert file to `diffusers` format:
		 - `python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path PATH_TO_CKPT_FILE --original_config_file PATH_TO_CONFIG_YAML_FILE --dump_path PATH_TO_OUTPUT_FOLDER --device DEVICE`
	 - Convert from `diffusers` format to ONNX format:
		 - `python convert_stable_diffusion_checkpoint_to_onnx.py --model_path PATH_TO_PREVIOUS_OUTPUT_FOLDER --output_path PATH_TO_ONNX_OUTPUT_FOLDER`
	 - All scripts work as expected.
 - Convert stable diffusion model from checkpoint file (`.ckpt` or `.safetensor`) via `diffusers` scripts and huggingface hub to onnx via command line:
	 - Clone the `diffusers` git repo:
		 - `git clone https://github.com/huggingface/diffusers.git`
	 - Convert file to `diffusers` format:
		 - `python ./diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path PATH_TO_CKPT_FILE --original_config_file PATH_TO_CONFIG_YAML_FILE --dump_path PATH_TO_OUTPUT_FOLDER --device DEVICE`
	 - Upload the converted model to Huggingface hub.
	 - Convert the model from Huggingface hub to onnx via `optimum`:
		 - `optimum-cli export onnx --model model_id/on_hub dest_folder`


### TODO

 - Working JS example/demo that uses SD 1.X and the exported ONNX model(s)
 - SD 2.0 workflow
 - SDXL workflow
 - LORA integration


### References

 - [Stable Diffusion NodeJS Repository from GitHub](https://github.com/dakenf/stable-diffusion-nodejs)
 - [Huggingface Hub Model from aislamov](https://huggingface.co/aislamov/stable-diffusion-2-1-base-onnx)
 - [Huggingface Documentation to Export/Run Inference Models/Stable Diffusion to ONNX](https://huggingface.co/docs/diffusers/optimization/onnx)
 - Manual conversion
	 - [YouTube video](https://www.youtube.com/watch?v=caCzBJcQ5jo&t=749s&ab_channel=Tech-Practice) on converting Civitai stable diffusion models model checkpoints (ckpt) to ONNX
	 - [Diffusers script](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) to convert a stable diffusion model (ckpt) to diffusers compatible format.
	 - [Diffusers script](https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py) to convert a stable diffusion model (diffusers) to onnx.
	 - Stable diffusion 1.0 [inference yaml file](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml) from CompVis repo (required for `convert_original_stable_diffusion_to_diffusers.py` from `diffusers/scripts/`).