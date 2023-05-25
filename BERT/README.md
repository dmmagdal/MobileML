# BERT Embeddings

Description: Download a BERT model from Tensorflow Hub in Python to create BERT embeddings and export it to Tensorflow JS. Then pass a string to the Tensorflow JS model in NodeJS program.


### Notes

 - Unable to properly convert BERT model from Tensorflow Hub to TFJS compatible format. Have tried the following: 
	 - Use python API for TensorflowJS converter. Was able to convert model but not load it in NodeJS (see video for possible explanation as to why it failed).
	 - Use TFJS converter script (model saved as H5 file). Was able to covert model but not load it in NodeJS.
	 - Used TFJS wizard converter script (TF Hub link). Was not able to convert model.
 - The main issue with converting the BERT model seems to possibly be issues with the input or output signatures not being defined for the model (regardless of conversion method).
	 - There is also this [GitHub issue](https://github.com/tensorflow/tfjs/issues/5734) tracking a similar issue.
 - Huggingface has limited support for Tensorflow models (including BERT) on their website. They seem to specialize in exporting models to ONNX format or TorchScript.
	 - Given that this repo isn't exclusively for Tensorflow/TensorflowJS only (though I'd personally prefer it), it could be worth an attempt to create a Huggingface feature-extracton pipeline to produce the same embedding outputs expected by the python BERT model from TensorflowHub and export the pipeline to ONNX. After all, the BERT model should be the same in terms of weights and architecture. 
	 - UPDATE: I've gone through the different ways to try and export a BERT model via huggingface to ONNX. This includes the following:
		 - Create custom pipeline.
		 - Use the BERTModel.
		 - Use a feature extraction pipeline. This one actually was able to work somewhat but it outputs the sentence outputs but we want the pooled outputs for this project.
	 - Other references:
		 - Microsoft [pytorch convert model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model)
		 - HuggingFace [custom pipeline](https://huggingface.co/docs/transformers/add_new_pipeline#share-your-pipeline-on-the-hub)
		 - Mendium [article](https://towardsdatascience.com/nlp-transformers-pipelines-with-onnx-9b890d015723)


### References

 - [Export Keras Model to TFJS](https://www.tensorflow.org/js/tutorials/conversion/import_keras)
 - [TFJS documentation](https://js.tensorflow.org/api/latest/)
 - [Pretrained BERT preprocess model in Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)
 - [Pretrained BERT model in Tensorflow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3)
 - [Initialize pretrained BERT model from Tensorflow Hub](https://github.com/dmmagdal/BERT_Database/blob/main/faiss_db/database_faiss.py)
 - [How to use Tensorflow JS converter](https://www.youtube.com/watch?v=yWBM2-Rx47M&ab_channel=OhYicong)