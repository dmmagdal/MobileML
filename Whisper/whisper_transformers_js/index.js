// Import modules.
// import {  } from '@xenova/transformers';
import { pipeline, env } from '@xenova/transformers';


async function  main () {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = './';

    // Initialize tokenizer & model.
    const model_id = 'whisper_base';
    // const model_id = 'dmmagdal/whisper-tiny-onnx';
    // const model_id = 'dmmagdal/whisper-small-onnx';
    // const model_id = 'dmmagdal/whisper-base-onnx';
    // const model_id = 'dmmagdal/whisper-medium-onnx';
    // const model_id = 'dmmagdal/whisper-large-onnx';
    const cache_dir = model_id.replace('dmmagdal/', '');
    const transcriber = await pipeline(
        'automatic-speech-recognition', 
        model_id,
        {
            quantized: false,        // passing in quantized: false means that transformers.js wont look for the quantized onnx model files
            cache_dir: cache_dir,    // passing in cache_dir value to specify where to save files locally.
            local_files_only: true,
        }
    );   // pipeline abstraction for all-in-one

    // Load the audio file and transcribe it.
    let file = '../jfk.wav';
    
    let output = await transcriber(
        file,
        {
            return_timestamps: true,    // whether to return timestamps (default is false -> no timestamps returned). Passing in 'word' will pass in word level timestamps.
            // language: 'english',     // the source language (default is null -> should be auto detected).
            // task: 'translation',     // task to perform (default is null -> should be auto detected). 'translation' means to translate to english. 'transcribe' means to transcribe to english.
            chunk_length_s: 30,         // the length of audio chunks to process in seconds (default is 0 -> no chunking).
            stride_length_s: 5,         // the length of overlap between consecutive audio chunks in seconds (default is chunk_length_s / 6).
        }
    )

    console.log(`Transcribed ${file}:`);
    console.log(JSON.stringify(output, null, '\t'));

    // Exit the program.
    process.exit(0);
}


main()