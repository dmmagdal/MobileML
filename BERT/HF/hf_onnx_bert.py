# hf_onnx_bert.py
# Initialize a custom pipeline for feature extraction using the BERT
# model on huggingface.
# Python 3.7
# Windows/MacOS/Linux


from pathlib import Path
import torch
from torch import onnx
from transformers import Pipeline, BertModel, BertTokenizer, pipeline
from transformers.pipelines import PIPELINE_REGISTRY
import transformers.convert_graph_to_onnx as onnx_convert


class BERTPipeline(Pipeline):
	def _sanitize_parameters(self, **kwargs):
		preprocess_kwargs = {}
		if "maybe_arg" in kwargs:
			preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
		return preprocess_kwargs, {}, {}

	def preprocess(self, inputs, maybe_arg=2):
		# model_input = torch.Tensor(inputs["input_ids"])
		model_input = self.tokenizer.encode(
			inputs, return_tensors='pt'
		)
		# return {"model_input": model_input}
		return model_input

	def _forward(self, model_inputs):
		# model_inputs == {"model_input": model_input}
		# outputs = self.model(**model_inputs)
		# Maybe {"logits": Tensor(...)}
		outputs = self.model(model_inputs)
		return outputs

	def postprocess(self, model_outputs):
		# best_class = model_outputs["logits"].softmax(-1)
		# return best_class
		return model_outputs[1]


def main():
	# Text inputs to be embedded by BERT.
	inputs = "There have always been ghosts in the machine. Random "+\
		"segments of code that when grouped together form unexpected "+\
		"protocols."

	# Initialize BERT tokenizer and model from pretrained model on
	# Huggingface Hub.
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# bert_model = BertModel(config)
	bert_model = BertModel.from_pretrained('bert-base-uncased')

	# Put model in "eval" mode.
	bert_model.eval()
	print("-" * 72)

	# -----------------------------------------------------------------
	# 'Vanilla' BERT model (no pipeline)
	# -----------------------------------------------------------------

	model_inputs = tokenizer.encode(inputs, return_tensors='pt') # tensor [b, len]
	output = bert_model(model_inputs)
	print(output)
	print(len(output))
	print(output[0].shape) # Outputs sentence embedding
	print(output[1].shape) # Outputs pooled embedding
	print("-" * 72)

	# -----------------------------------------------------------------
	# 'Vanilla' BERT feature extraction pipeline
	# -----------------------------------------------------------------

	# Valid feature extraction code. Note that it only outputs sentence
	# embeddings though.
	bert_pipeline = pipeline(
		'feature-extraction', model='bert-base-uncased'
	)

	print("Output from vanilla BERT pipeline:")
	output = bert_pipeline(inputs, return_tensors=True)
	print(output)
	print(len(output))
	print(output[0].shape) # Outputs sentence embedding
	print("-" * 72)

	# -----------------------------------------------------------------
	# Custom BERT feature extraction pipeline
	# -----------------------------------------------------------------

	# Initialize and register custom pipeline for bert feature
	# extraction.
	PIPELINE_REGISTRY.register_pipeline(
		"bert-feature-extraction",
		pipeline_class=BERTPipeline,
		# pt_model=bert_model,
		# default={"pt": ("user/awesome_model", "abcdef")},
		# type="text",  # current support type: text, audio, image, multimodal
	)

	bert_embeddings_pipeline = pipeline(
		'bert-feature-extraction', tokenizer=tokenizer,
		# model="bert-base-uncased"
		model=bert_model # Use explicitly defined model, not str for pretrained model (ie )
	)

	print("Output from custom BERT pipeline:")
	output = bert_embeddings_pipeline(inputs, return_tensors=True)
	print(output)
	print(len(output))
	print(output[0].shape) # Outputs pooled embedding
	print("-" * 72)

	# Export the pipeline to ONNX format.
	bert_model.to('cpu')
	onnx_convert.convert_pytorch(
		bert_embeddings_pipeline, 
		opset=11, # Use newest operators that are supported
		output=Path("bert.onnx"), # use Path from pathlib, not raw str
		use_external_format=False
	)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()