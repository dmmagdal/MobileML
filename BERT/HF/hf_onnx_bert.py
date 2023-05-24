# hf_onnx_bert.py
# Initialize a custom pipeline for feature extraction using the BERT
# model on huggingface.
# Python 3.7
# Windows/MacOS/Linux


import onnx
import torch
from torch import onnx
from transformers import Pipeline, BertModel, BertTokenizer, pipeline
from transformers.pipelines import PIPELINE_REGISTRY


class BERTPipeline(Pipeline):
	def _sanitize_parameters(self, **kwargs):
		preprocess_kwargs = {}
		if "maybe_arg" in kwargs:
			preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
		return preprocess_kwargs, {}, {}

	def preprocess(self, inputs, maybe_arg=2):
		# model_input = torch.Tensor(inputs["input_ids"])
		# model_input = self.tokenizer.encode(inputs)['input_ids']
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

	# Initialize and register pipeline.
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# bert_model = BertModel(config)
	bert_model = BertModel.from_pretrained('bert-base-uncased')

	'''
	PIPELINE_REGISTRY.register_pipeline(
		"bert-feature-extraction",
		pipeline_class=BERTPipeline,
		# pt_model=bert_model,
		# default={"pt": ("user/awesome_model", "abcdef")},
		# type="text",  # current support type: text, audio, image, multimodal
	)

	bert_embeddings_pipeline = pipeline(
		'bert-feature-extraction', tokenizer=tokenizer,
		model="bert-base-uncased"
	)
	'''

	bert_model.eval()

	# Valid feature extraction code.
	# bert_pipeline = pipeline(
	# 	'feature-extraction', model='bert-base-uncased'
	# )

	inputs = "There have always been ghosts in the machine. Random "+\
		"segments of code that when grouped together form unexpected "+\
		"protocols."
	
	# output = bert_pipeline(inputs, return_tensors=True)
	# output = bert_embeddings_pipeline(inputs , return_tensors=True)
	model_inputs = tokenizer.encode(inputs, return_tensors='pt')
	output = bert_model(model_inputs)
	print(output)
	print(len(output))
	print(output[0].shape) # Outputs sentence embedding


	onnx_path = "./bert_model.onnx"
	torch.onnx.export(
		model=bert_model,
		args=tuple(model_inputs.values()),
		f=onnx_path,
		input_names=list(model_inputs.keys()),
		output_names=["pooled_output"],
		opset_version=11,
		dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
		verbose=True
	)

	# Load the exported ONNX model
	onnx_model = onnx.load(onnx_path)

	# Verify the ONNX model
	onnx.checker.check_model(onnx_model)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()