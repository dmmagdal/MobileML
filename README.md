# Mobile ML

Description: A collection of public works and examples that bring machine learning to mobile and web platforms.


### Notes:
 - TensorflowJS
     - TFJS runs on WebGL in browser ([reference](https://www.tensorflow.org/js/guide/layers_for_keras_users#memory_management_of_layer_and_model_objects))
         - The weights of Model and Layer objects are backed by WebGL textures.
         - WebGL has no built in garbage collection, use the `dispose()` function to dispose of a Model or Layer object.
         - WebGL is executed/run on a device's GPU ([reference](https://en.wikipedia.org/wiki/WebGL#:~:text=WebGL%20programs%20consist%20of%20control,the%20non%2Dprofit%20Khronos%20Group.))
         - For more information on TensorflowJS and how it leverages hardware, see [here](https://www.tensorflow.org/js/guide/platform_environment#shader_compilation_texture_uploads)
 - Huggingface & Pytorch
 - ONNX Runtime


### References:
 - TensorflowJS
     - [NPM page](https://www.npmjs.com/package/@tensorflow/tfjs)
     - GitHub
         - [Pretrained Models](https://github.com/tensorflow/tfjs-models)
     - [Tensorflow JS main page](https://www.tensorflow.org/js)
     - [Getting Started/Tutorials page](https://www.tensorflow.org/js/tutorials)
     - [Latest TensorflowJS Core Documentations](https://js.tensorflow.org/api/latest/)
     - [Guide page](https://www.tensorflow.org/js/guide)
     - [TensorflowJS layers API for Keras users](https://www.tensorflow.org/js/guide/layers_for_keras_users) (Good for Keras users to get familiar with TFJS syntax)
     - [TensorflowJS models and layers](https://www.tensorflow.org/js/guide/models_and_layers)
 - Ndarray
     - [NPM page](https://www.npmjs.com/package/ndarray)
 - Huggingface
     - [Huggignface JS libraries](https://huggingface.co/docs/huggingface.js/index)
     - [Huggingface hub API (JS)](https://huggingface.co/docs/huggingface.js/hub/README)
     - [Huggingface inference API (JS)](https://huggingface.co/docs/huggingface.js/inference/README)
 - Brain.js
     - [NPM page](https://www.npmjs.com/package/brain.js)
