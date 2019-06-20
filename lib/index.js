const tf = require('@tensorflow/tfjs-node-gpu');
const wav = require('node-wav');
const fs = require('fs');

const SHAPE = 512;

async function main(data) {
  const model = await tf.loadLayersModel('file://kickPitchDetector/model.json');
  return pred(data, model);
}

function pred(data, model) {
  let xs = load(data);

  const preds = model.predict(xs).dataSync();

  return indexToTone(preds);
}

function load(data) {
  let buffer = (typeof data === 'object') ? data : fs.readFileSync(path);
  let result = wav.decode(buffer);
  let channelData = result.channelData[0];

  let dataArray = new Float32Array(SHAPE);
  for (let i = 0; i < SHAPE; i++) {
    let index = Math.ceil(i * channelData.length / SHAPE);
    dataArray[i] = channelData[index];
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

module.exports = main;
