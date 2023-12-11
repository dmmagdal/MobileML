# Whisper

Description: Export whisper models from huggingface to ONNX, then run the model for inference in a NodeJS program.


### Setup

 - It is recommended that you create a python virtual environment instead of a conda due to version issues with a lot of necessary packages.
 - To set up the virtual environment, install the venv package:
	 - `pip3 install virtualenv`
 - Create the new virtual environment:
	 - `python -m venv whisper-env`
 - Activate the virtual environment:
	 - Linux/MacOS: `source autogptq-env/bin/activate`
	 - Windows: `.\whisper-env\Scripts\activate`
 - Deactivate the virtual environment:
	 - `deactivate`
 - Install the necessary packages (while the virtual environment is active):
	 - `(whisper-env) pip3 install -r requirements.txt`
 - Also be sure to install the necessary version of pytorch according to your OS (refer to the pytorch website but the following command will help):
 - Linux & Windows (CUDA 11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
 - MacOS: `pip3 install torch torchvision torchaudio`


### Notes

 - Convert stable diffusion model from huggingface hub to onnx via command line:
	 - `optimum-cli export onnx --model model_id/on_hub dest_folder`
 - Original Whisper model checkpoints (from Huggingface hub):
     - [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
     - [openai/whisper-small](https://huggingface.co/openai/whisper-small)
     - [openai/whisper-base](https://huggingface.co/openai/whisper-base)
     - [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)
     - [openai/whisper-large](https://huggingface.co/openai/whisper-large)
 - ONNX converted Whisper model checkpoints:
     - [dmmagdal/whisper-tiny-onnx](https://huggingface.co/dmmagdal/whisper-tiny-onnx)
     - [dmmagdal/whisper-small-onnx](https://huggingface.co/dmmagdal/whisper-small-onnx)
     - [dmmagdal/whisper-base-onnx](https://huggingface.co/dmmagdal/whisper-base-onnx)
     - [dmmagdal/whisper-medium-onnx](https://huggingface.co/dmmagdal/whisper-medium-onnx)
     - [dmmagdal/whisper-large-onnx](https://huggingface.co/dmmagdal/whisper-large-onnx)


### References

 - [Huggingface Documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model): Export a model to ONNX with optimum.exporters.onnx
 - [Transformers JS](https://huggingface.co/docs/transformers.js/tutorials/node): Server-side inference in NodeJS
 - [Transformers JS](https://huggingface.co/docs/transformers.js/pipelines): The pipeline API
 - [Transformers JS](https://huggingface.co/docs/transformers.js/api/pipelines#pipelinesautomaticspeechrecognitionpipeline): ASR pipeline
