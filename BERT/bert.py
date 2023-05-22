# bert.py
# Initialize a BERT model from tensorflow hub and convert it to a
# format for tensorflow JS.
# Windows/MacOS/Linux
# Python 3.7
# Tensorflow 2.7.0


import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflowjs as tfjs
import tensorflow_text as text


def build_bert():
	# Tensorflow Hub links for the BERT preprocessor and encoder model.
	BERT_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3" # this is marked as a text embedding model
	PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" # this is marked as a text preprocessing model

	# Create a BERT model with the Keras Functional API.
	text_input = keras.layers.Input(shape=(), dtype=tf.string)
	preprocessor_layer = hub.KerasLayer(PREPROCESS_MODEL)
	encoder_inputs = preprocessor_layer(text_input)
	encoder = hub.KerasLayer(BERT_MODEL, trainable=False)
	outputs = encoder(encoder_inputs)
	pooled_output = outputs["pooled_output"] # [batch_size, 768] (represent each input sequence as a whole)
	sequence_output = outputs["sequence_output"] # [batch_size, seq_length, 768] (represent each input token in context)
	model = keras.Model(
		inputs=text_input, outputs=pooled_output, name="Bert"
	)
	model.trainable = False
	model.summary()

	# Save the model.
	# model.save(self.save_path)

	# Return the model.
	return model


def main():
	# Initialize BERT model.
	model = build_bert()

	# Convert model to tensorflowJS.
	model_save = "./tfjs_bert"
	tfjs.converters.save_keras_model(model, model_save)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()