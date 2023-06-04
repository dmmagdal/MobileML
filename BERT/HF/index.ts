// Load the exported BERT model (onnx format) and pass it some text to
// embed.

// import { loadTokenizer } from './bert_tokenizer.ts';
// import * as wasmFeatureDetect from 'wasm-feature-detect';
// import ort from 'onnxruntime-web';
import ort from 'onnxruntime-node';
// import * as tf from '@tensorflow/tfjs';
import { AutoTokenizer} from '@xenova/transformers';
import fs from 'fs';
import path from 'path';
import process from 'process';




// Inputs.
const inputs = '\
 There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';

// const model = new URL('../plain_bert.onnx', import.meta.url).pathname;
// const model = path.join(__dirname, '..', 'plain_bert.onnx');
const model = path.join(process.cwd(), '..', 'plain_bert.onnx');
// const model = path.join(process.cwd(), '..', 'bert.onnx');
// const model: string = '../bert.onnx';
// const model = './bert_Int8.onnx';

// console.log(process.cwd())
// console.log(fs.existsSync(model))
// process.exit()

// Initialize inference session with ort.
const session = await ort.InferenceSession.create(
  model, 
  // '../../bert.onnx',
  {
    // executionProviders: ['wasm'], // can also specify 'webgl'
    // executionProviders: ['webgl'], // can also specify 'wasm'
    graphOptimizationLevel: 'all'
  }
);
// const session = await new ort.InferenceSession();
// await session.loadModel(
//   model, 
//   {
//     executionProviders: ['wasm'], // can also specify 'webgl
//     graphOptimizationLevel: 'all'
//   }
// )

// Inputs.
const inputs_ = 'There have always been ghosts in the machine. Random\
 segments of code that when grouped together form unexpected\
 protocols.\
';

const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
const tokenized_inputs = tokenizer(inputs);
// const tokenized_inputs = tokenizer(inputs, {padding: true, max_length: 512});
// const tokenized_inputs = tokenizer._call(
//   inputs, 
//   {
//     padding: true, // whether to pad input
//     max_length: 512 // max input length (where to pad to if applicable)
//   }
// );
console.log(tokenized_inputs)

const input_tensor = new ort.Tensor(
  'int64', tokenized_inputs.input_ids.data, tokenized_inputs.input_ids.dims
);


const data = new BigInt64Array(512);
const output = await session.run(
  {
    'input_ids': input_tensor
    // 'input.1': new ort.Tensor('int64', data, [1, 512])//.setValues(data)
    // 'input_ids': new ort.Tensor('int64', data, [1, 512])//.setValues(data)
  }
  // [
  //   new ort.Tensor('int64', [512])//.setValues(data)
  // ]
);

console.log(output);


process.exit();


async function inference(session: ort.InferenceSession, preprocessedData: any): Promise<void> {
  // Get start time to calculate inference time.
  const start = new Date();

  // create feeds with the input name from model export and the preprocessed data.
  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = preprocessedData;
  
  // Run the session inference.
  const outputData = await session.run(feeds);

  // Get the end time to calculate inference time.
  const end = new Date();
}




/*
const options = {
  executionProviders: ['wasm'], // can also specify 'webgl
  graphOptimizationLevel: 'all'
};


let downLoadingModel = true;
const model = './bert.onnx';
// const model = './bert_Int8.onnx';

const session = ort.InferenceSession.create(model, options);
session.then(t => { 
  downLoadingModel = false;
  //warmup the VM
  for(var i = 0; i < 10; i++) {
    console.log("Inference warmup " + i);
    lm_inference("this is a warmup inference");
  }
});

const tokenizer = loadTokenizer()




function create_model_input(encoded: Array[int]) {
  let input_ids = new Array(encoded.length+2);
  let attention_mask = new Array(encoded.length+2);
  let token_type_ids = new Array(encoded.length+2);
  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  let i = 0;
  for(; i < encoded.length; i++) { 
    input_ids[i+1] = BigInt(encoded[i]);
    attention_mask[i+1] = BigInt(1);
    token_type_ids[i+1] = BigInt(0);
  }
  input_ids[i+1] = BigInt(102);
  attention_mask[i+1] = BigInt(1);
  token_type_ids[i+1] = BigInt(0);
  const sequence_length = input_ids.length;
  input_ids = new ort.Tensor('int64', BigInt64Array.from(input_ids), [1,sequence_length]);
  attention_mask = new ort.Tensor('int64', BigInt64Array.from(attention_mask), [1,sequence_length]);
  token_type_ids = new ort.Tensor('int64', BigInt64Array.from(token_type_ids), [1,sequence_length]);
  return {
    input_ids: input_ids,
    attention_mask: attention_mask,
    token_type_ids:token_type_ids
  }
}

async function lm_inference(text: string) {
  try { 
    const encoded_ids = await tokenizer.then(t => {
      return t.tokenize(text); 
    });
    if(encoded_ids.length === 0) {
      return [0.0, EMOJI_DEFAULT_DISPLAY];
    }
    const start = performance.now();
    const model_input = create_model_input(encoded_ids);
    const output =  await session.then(s => { return s.run(model_input,['output_0'])});
    const duration = (performance.now() - start).toFixed(1);
    const probs = output['output_0'].data.map(sigmoid).map( t => Math.floor(t*100));
    
    const result = [];
    for(var i = 0; i < EMOJIS.length;i++) {
      const t = [EMOJIS[i], probs[i]];
      result[i] = t;
    }
    result.sort(sortResult); 
    
    const result_list = [];
    result_list[0] = ["Emotion", "Score"];
    for(i = 0; i < 6; i++) {
       result_list[i+1] = result[i];
    }
    return [duration,result_list];    
  } catch (e) {
    return [0.0, EMOJI_DEFAULT_DISPLAY];
  }
}    
// export let inference = lm_inference 
// export let columnNames = EMOJI_DEFAULT_DISPLAY
// export let modelDownloadInProgress = isDownloading
*/



// const input_tensor = tf.cast(tf.tensor([inputs]), 'string');

// Output from model.

// Print out to console.
// console.log('Input:', inputs);
// console.log('Input BERT Embedding:', output_tensor);