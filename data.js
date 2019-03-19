const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const wav = require('node-wav');

const BATCH = 90;
const SHAPE = 1024;

module.exports = class Data {
  _load(path) {
    let buffer = fs.readFileSync(path);
    let result = wav.decode(buffer);
    let data = result.channelData[0];

    let dataArray = new Float32Array(SHAPE);
    for (let i = 0; i < SHAPE; i++) {
      let index = Math.ceil(i * data.length / SHAPE);
      dataArray[i] = data[index];
    }

    return dataArray;
  }

  nextTrainBatch() {
    let sounds = [];
    let pitches = [];

    for (let i = 0; i < BATCH; i++) {
      let soundLabel = getRandomInt(5) + 1;
      pitches.push(getRandomInt(23));
      sounds.push(this._load(`./samples/train/${soundLabel}_${pitches[i] + 1}.wav`));
    }

    return this.nextBatch(sounds, pitches);
  }

  nextValidBatch() {
    let sounds = [];
    let pitches = [];

    for (let i = 0; i < BATCH / 5; i++) {
      let soundLabel = getRandomInt(5) + 1;
      pitches.push(getRandomInt(23));
      sounds.push(this._load(`./samples/validation/${soundLabel}_${pitches[i] + 1}.wav`));
    }

    return this.nextBatch(sounds, pitches);
  }

  nextTestBatch() {
    let sounds = [];
    let pitches = [1];

    for (let i = 0; i < 1; i++) {
      // let soundLabel = getRandomInt(5) + 1;
      // pitches.push(getRandomInt(23));
      sounds.push(this._load(`./samples/test/1.wav`));
    }

    return this.nextBatch(sounds, pitches);
  }

  nextBatch(sounds, pitchLabels) {
    let labels = [];
    for (let i = 0; i < sounds.length; i++) {
      let array = new Array(24);
      array.fill(0);
      let index = pitchLabels[i];
      array[index] = 1;
      labels.push(array);
    }

    let xs = tf.tensor(sounds, [sounds.length, SHAPE]);
    let label = tf.tensor2d(labels, [sounds.length, 24]);

    return { xs, label };
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}
