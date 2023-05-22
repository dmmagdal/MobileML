// Load the exported TensorflowJS BERT model and pass it some text to
// embed.


import * as tf from '@tensorflow/tfjs';
import fs from 'fs';
import path from 'path';
import process from 'process';


// Load model with TFJS.
console.log(path.join(process.cwd(), 'tfjs_bert', 'model.json'));
console.log(
  fs.existsSync(path.join(process.cwd(), 'tfjs_bert', 'model.json'))
);
const model = await tf.loadLayersModel(
  path.join(process.cwd(), 'tfjs_bert', 'model.json')
);
model.summary();
// const model = await tf.loadGraphModel(
//   path.join(process.cwd(), 'tfjs_bert', 'model.json')
// );

// Inputs.
const inputs = '\
  There have always been ghosts in the machine. Random \
  segments of code that when grouped together form unexpected \
  protocols.\
';
const input_tensor = tf.cast(tf.tensor([inputs]), 'string');

// Output from model.
const output_tensor = model.predict(input_tensor);

// Print out to console.
console.log('Input:', inputs);
console.log('Input BERT Embedding:', output_tensor);