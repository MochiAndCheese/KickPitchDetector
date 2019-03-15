const tf = require('@tensorflow/tfjs-node-gpu');
const wav = require('node-wav');
const fs = require('fs');

const SHAPE = 512;

async function main() {
  const model = await tf.loadLayersModel('file://./kickPitchDetector/model.json');
  let path = process.argv[2];
  if (typeof path !== 'string') throw new Error('Argument is not a string.');

  pred(path, model);
}

function pred(path, model) {
  let xs = load(path);

  const preds = model.predict(xs).dataSync();

  console.log(indexToTone(preds));
}

function load(path) {
  let buffer = fs.readFileSync(path);
  let result = wav.decode(buffer);
  let data = result.channelData[0];

  let dataArray = new Float32Array(SHAPE);
  for (let i = 0; i < SHAPE; i++) {
    let index = Math.ceil(i * data.length / SHAPE);
    dataArray[i] = data[index];
  }

  return tf.tensor(dataArray, [1, SHAPE]);
}

function indexToTone(preds) {
  let max = Math.max(...preds);
  let pitch = preds.indexOf(max);

  const tones = [
    'C4', 'B3', 'A#3', 'A3', 'G#3', 'G3', 'F#3', 'F3', 'E3', 'D#3', 'D3', 'C#3',
    'C3', 'B2', 'A#2', 'A2', 'G#2', 'G2', 'F#2', 'F2', 'E2', 'D#2', 'D2', 'C#2'
  ]

  return tones[pitch];
}

main();
