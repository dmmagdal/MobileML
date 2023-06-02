// Load the BERT model from huggingface transformers js and pass it 
// some text to embed.

import { AutoModel, AutoTokenizer, pipeline } from '@xenova/transformers';


// Inputs.
const input_text = '\
 There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';

// Initialize model and tokenizer.
let tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
let model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');

// Pass inputs through to the tokenizer and model.
let inputs = tokenizer(input_text);
let { logits } = await model(inputs);
// Tensor {
//     data: Float32Array(702006) [-7.117443084716797, -7.107812881469727, -7.092104911804199, ...]
//     dims: (3) [1, 23, 30522],
//     type: "float32",
//     size: 702006,
// }

console.log(logits);

// Perform the same operation as above, only this time using the
// pretrained model's _call() function. The results should be exactly
// the same as above.
let call_outputs = await model._call(inputs);
console.log(call_outputs.logits);

// Initialize a (feature extraction) pipeline. Pass the input through
// the pipeline for processing.
// Reference: https://huggingface.co/docs/transformers.js/api/pipelines#pipelinesfeatureextractionpipeline-codepipelinecode
let pipe = await pipeline(
  'feature-extraction', // task
  'Xenova/bert-base-uncased', // pretrained model
  {revision: 'default'}, // pretrained options.
);
let output = await pipe(
  input_text, 
  // {pooling: 'mean', normalize: true} // pool & normalize logits from BERT.
  {pooling: 'mean', normalize: false} // pool & normalize logits from BERT.
);
console.log(output);

// Print out to console.
console.log('Input:', input_text);
console.log('BERT Embedding:', output);