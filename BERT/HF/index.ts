// Load the exported BERT model (onnx format) and pass it some text to
// embed.

// import ort from 'onnxruntime-web';
import ort from 'onnxruntime-node';
import { AutoTokenizer } from '@xenova/transformers';
import path from 'path';
import process from 'process';


// Inputs.
const inputs = '\
 There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';

// Model paths.
const model_ = path.join(process.cwd(), '..', 'plain_bert.onnx'); // Exported BERT
const model_pipe = path.join(process.cwd(), '..', 'bert.onnx'); // Exported BERT pipeline

// Initialize inference session with ort.
const session = await ort.InferenceSession.create( // Exported BERT
  model_,
  {
    // executionProviders: ['wasm'], // can also specify 'webgl'
    // executionProviders: ['webgl'], // can also specify 'wasm'
    graphOptimizationLevel: 'all',
  }
);
const session_pipe = await ort.InferenceSession.create( // Exported BERT pipeline
  model_pipe,
  {
    // executionProviders: ['wasm'], // can also specify 'webgl'
    // executionProviders: ['webgl'], // can also specify 'wasm'
    graphOptimizationLevel: 'all',
  }
);

// Initialize tokenizer with transformers-js and tokenize the inputs.
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
const tokenized_inputs = tokenizer(inputs);
console.log(tokenized_inputs)

// Send tokenized inputs to (ORT) tensor.
const input_tensor = new ort.Tensor(
  'int64', // Tensor dtype
  tokenized_inputs.input_ids.data, // Tensor data
  tokenized_inputs.input_ids.dims // Tensor dims/shape
);
const attention_tensor = new ort.Tensor( // Required for exported BERT pipeline
  'int64',
  tokenized_inputs.attention_mask.data,
  tokenized_inputs.attention_mask.dims
);
const token_ids_tensor = new ort.Tensor( // Required for exported BERT pipeline
  'int64',
  tokenized_inputs.token_type_ids.data,
  tokenized_inputs.token_type_ids.dims
);

// Output from model.
const output = await session.run(
  {
    'input_ids': input_tensor, // Only required named input (see export function for BERT model in hf_onnx_bert.py)
    // 'attention_mask': attention_tensor, // Extra named inputs are ignored. Only required ones will be read
    // 'token_type_ids': token_ids_tensor, // Extra named inputs are ignored. Only required ones will be read
  }
);
const output_pipeline = await session_pipe.run(
  {
    'input_ids': input_tensor,
    'attention_mask': attention_tensor,
    'token_type_ids': token_ids_tensor,
  }
);

console.log(output);
console.log(output_pipeline);

// Print out to console.
console.log('Input:', inputs);
console.log('BERT Embedding (model):', output['1740']); // Specify pooled outputs (exported BERT)
console.log('BERT Embedding (pipeline):', output_pipeline['output_1']); // Specify pooled outputs (exported BERT pipeline)