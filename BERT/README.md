# BERT Embeddings

Description: Download a BERT model from Tensorflow Hub in Python to create BERT embeddings and export it to Tensorflow JS. Then pass a string to the Tensorflow JS model in NodeJS program.


### Notes

 - Unable to properly convert BERT model from Tensorflow Hub to TFJS compatible format. Have tried the following: 
	 - Use python API for TensorflowJS converter. Was able to convert model but not load it in NodeJS (see video for possible explanation as to why it failed).
	 - Use TFJS converter script (model saved as H5 file). Was able to covert model but not load it in NodeJS.
	 - Used TFJS wizard converter script (TF Hub link). Was not able to convert model.
 - The main issue with converting the BERT model to a SavedModel or TensorflowJS format seems to possibly be issues with the input or output signatures not being defined for the model (regardless of conversion method).
	 - There is also this [GitHub issue](https://github.com/tensorflow/tfjs/issues/5734) tracking a similar issue.
 - Huggingface has limited support for Tensorflow models (including BERT) on their website. They seem to have more models developed in pytorch instead. For those models, Huggingface provides ways to export those models to ONNX format or TorchScript.
	 - Given that this repo isn't exclusively for Tensorflow/TensorflowJS only (though I'd personally prefer it), it could be worth an attempt to create a Huggingface feature-extracton pipeline to produce the same embedding outputs expected by the python BERT model from TensorflowHub and export the pipeline to ONNX. After all, the BERT model should be the same in terms of weights and architecture. 
	 - UPDATE: I've gone through the different ways to try and export a BERT model via huggingface to ONNX. This includes the following:
		 - Use the BERTModel. The way to export the pytorch BERT model from huggingface seemed very complicated. I was able to export a model to an `.onnx` file though. 
		 - Use a feature extraction pipeline. This one actually was able to work somewhat but it outputs the sentence outputs but we want the `pooled outputs` for this project.
		 - Create custom pipeline. This actually worked out well for the program and was able to export the model (with only the `pooled outputs` as the pipeline output) to an `.onnx` file.
	 - References for the above:
		 - Microsoft [pytorch convert model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model)
		 - HuggingFace [custom pipeline](https://huggingface.co/docs/transformers/add_new_pipeline#share-your-pipeline-on-the-hub)
	 - Additional references (these may not have been implemented but are still helpful)
		 - Medium [article](https://towardsdatascience.com/nlp-transformers-pipelines-with-onnx-9b890d015723)
		 - Huggingface serialization [export to onnx][https://huggingface.co/docs/transformers/serialization]
		 - Huggingface convert transformers to onnx with huggingface optimum [article](https://huggingface.co/blog/convert-transformers-to-onnx)
		 - Converting models to ONNX [YouTube](https://www.youtube.com/watch?v=lRBsmnBE9ZA)
	 - When converting a model to ONNX (especially text models like BERT, T5, or GPT2), be sure to specify the `dynamic_axes` argument in the `torch.onn.export()` function. That argument lets ONNX know that the specified inputs/outputs are expected to be dynamic. By default, no dictionary of input/output names are specified which means that the model will expect all inputs and outputs sizes/shapes to match those given in the args. See the [documentation](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export) for more details on the function.
		 - A quick note on padding inputs. Tokenizers in Transformers-JS allow for inputs to be padded up to a max length. Padding does affect the BERT embedding values of an input (e.g. a sentence without padding from the tokenizer will embed to a differet value compared to a sentence with padding). In the Python module of Huggingface Transformers (and Tensorflow), inputs are not padded automatically. It is for this reason why using dynamic axes is very important.
	 - Using the ONNX runtime in javascript has a few odd caveats:
		 - The `onnxruntime-web` node module is good for running ONNX models in web browser. The `onnxruntime-node` node module is for running ONNX mdoels in NodeJS in the backend. It is recommended that `onnxruntime-node` is used, as I intend to run models in both electron & react native apps for desktop and mobile devices respectively. 
		 - The documentation for onnxruntime -web (or -node) is sparse at best. Will have do to a lot of trial and error.
		 - ONNX runtime runs models through an `InferenceSession` object. The path to the ONNX model is supplied to the `create()` function as part of the standard arguments. Additional arguments can be specified in an object such as what backend to use (webgl or wasm). 
			 - I could not get wasm or webgl backends to run on my machine so I decided to leave that object undefined/empty.
			 - To pass in data to a model, use the `run()` function from the `InferenceSession` object. The data is expected to be in an object format where the keys match the expected input names for the exported ONNX model and the values are tensors from the ONNXruntime library (ensure correct shape and dtype).
 - There is also a node library that is linked or endorsed by Huggingface called transformers-js.
	 - Transformers-js uses ONNX runtime to run models in browser, therefore it only works with models in ONNX format. Models can be converted to ONNX format via torch.onnx.export() (the native pytorch way), transformers.convert_graph_to_onnx (valid with Huggingface v4.24.0), or Optimum (a new Huggingface module designed to work with Transformers to export models to ONNX; see [here](https://huggingface.co/blog/convert-transformers-to-onnx)).
	 - Transformers-JS comes with a wide range of support for [pipelines](https://huggingface.co/docs/transformers.js/pipelines), [tokenizers](https://huggingface.co/docs/transformers.js/api/tokenizers), and [models](https://huggingface.co/docs/transformers.js/api/models) for different tasks. Tokenizers and models can be initialized from pretrained sources from the Huggingface model hub.
	 - For the BERT example, it using BERT from Transformers-JS only gives the logits from the model, not the usual sentence output & pooled output. The logits are the output of the BERT Model before a softmax activation function is applied to the output of BERT ([source](https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209)). Retrieving the sentence output and pooled output requires more work.
		 - Using the BERT feature extraction pipeline in Transformers-JS, adding the pooling & normalization function applies the function specified in the `pooling` argument. Normalization is also applied the same way via specifying the `normalization` function. This still only produces the logits from the model. 
	 - As mentioned before, tokenizers in Transformers-JS provides the ability to pad inputs. This is done via setting the `padding` option to true and `max_length` option to a value (see tokenizers [_call() documentation](https://huggingface.co/docs/transformers.js/api/tokenizers#pretrainedtokenizercalltext-options-codeobjectcode)).
 - Apple Silicon caveats
	 - Unable to run onnx and onnxruntime python modules for model quantization. Could be due to incompatible version of modules on instance or that the modules don't place nice with Macbook M1 silicon.
	 - Building anything Tensorflow or PyTorch related on Docker will lead to a core dump at runtime (if the image is even able to be built in the first place).
 - I ended up doing the following to get BERT to run in javascript:
	 - Use Transformers-JS for the pretrained tokenizer rather than implement it on my own.
	 - Export the BERT model from Huggingface to ONNX with `torch.onnx.export()`, be sure to specify `dynamic_axes` for the inputsl.
	 - Use ONNX Runtime Node to create an inference session and pass tokenized data to the model. The output for BERT will be both the sentence output and pooled output as expected.


### TODO List

 [ ] Fix Dockerfile for exporting BERT with Huggingface. Currently unable to build docker image from Dockerfile on Macbook.


### References

 - Tensorflow (and Tensorflow JS)
	 - [Export Keras Model to TFJS](https://www.tensorflow.org/js/tutorials/conversion/import_keras)
	 - [TFJS documentation](https://js.tensorflow.org/api/latest/)
	 - [Pretrained BERT preprocess model in Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)
	 - [Pretrained BERT model in Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3)
	 - [Initialize pretrained BERT model from Tensorflow Hub](https://github.com/dmmagdal/BERT_Database/blob/main/faiss_db/database_faiss.py)
	 - [How to use Tensorflow JS converter](https://www.youtube.com/watch?v=yWBM2-Rx47M&ab_channel=OhYicong)
 - Huggingface (to ONNX)
	 - [Huggingface Transformers: How to create a custom pipeline Tutorial](https://huggingface.co/docs/transformers/add_new_pipeline)
	 - [Deploy Transformer Models in the Browser with ONNXRuntime YouTube](https://www.youtube.com/watch?v=W_lUGPMW_Eg)
	 - [Inference in Javascript with ONNX Runtime Web YouTube](https://www.youtube.com/watch?v=vYzWrT3A7wQ&ab_channel=ONNXRuntime)
	 - [How to run PyTorch Models in the Browser with ONNX.js](https://www.youtube.com/watch?v=Vs730jsRgO8)
	 - [PyTorch Documentation: ONNX.export()](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export)
 - ONNX Runtime
	 - [ONNX Runtime Repository from GitHub](https://github.com/microsoft/onnxruntime)
	 - [ONNX Runtime Web NPM Module](https://www.npmjs.com/package/onnxruntime-web)
	 - [ONNX Runtime Node NPM Module](https://www.npmjs.com/package/onnxruntime-node)
	 - [ONNX Runtime Documentation Home Page](https://onnxruntime.ai/docs/)
	 - [ONNX Runtime for Javascript Documentation](https://onnxruntime.ai/docs/get-started/with-javascript.html)
	 - [ONNX Runtime Documentation: Deploy on Web](https://onnxruntime.ai/docs/tutorials/web/)
	 - [Inference PyTorch BERT Model with ONNX Runtime on CPU](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb)
	 - [ONNX Inference Runtime Example: Quick Start to ONNX Runtime Node](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/quick-start_onnxruntime-node/index.js)
	 - [ONNX Inference Runtime Example: Inference Session](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/api-usage_inference-session/inference-session-run.js)
 - Huggingface Transformers JS
	 - [Transformers-JS Repository from GitHub](https://github.com/xenova/transformers.js)
	 - [Transformers-JS NPM Module](https://www.npmjs.com/package/@xenova/transformers)
	 - [Transformers-JS Documentation Page on Huggingface Website](https://huggingface.co/docs/transformers.js/index)
	 - [Transformers-JS Documentation: The Pipeline API](https://huggingface.co/docs/transformers.js/pipelines)
	 - [Transformers-JS Documentation: Pipelines](https://huggingface.co/docs/transformers.js/api/pipelines)
	 - [Transformers-JS Documentation: Pipelines - Feature Extraction Pipeline](https://huggingface.co/docs/transformers.js/api/pipelines#pipelinesfeatureextractionpipeline-codepipelinecode)
	 - [Transformers-JS Documentation: Tokenizers](https://huggingface.co/docs/transformers.js/api/tokenizers)
	 - [Transformers-JS Documentation: Tokenizer call Options](https://huggingface.co/docs/transformers.js/api/tokenizers#pretrainedtokenizercalltext-options-codeobjectcode)
	 - [Huggingface Hub Models from Xenova](https://huggingface.co/models?sort=downloads&search=xenova)