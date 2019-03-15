const Data = require('./data.js');
const tf = require('@tensorflow/tfjs-node-gpu');
const data = new Data();

const TEST_BATCH = 1;

function main() {
  tf.tidy(async () => {
    const model = await tf.loadLayersModel('file://./kickPitchDetector/model.json');
    await test(model);
  });
}

async function test(model) {
  for (let TEST_COUNT = 0; TEST_COUNT < TEST_BATCH; TEST_COUNT++) {
    let { xs, label } = await data.nextTestBatch();

    // console.log(`Test[${TEST_COUNT + 1} / ${TEST_BATCH}]`);

    const preds = model.predict(xs).dataSync();
    let max = Math.max(...preds);
    let pitch = preds.indexOf(max);
    console.log(pitch + 1);
  }
}

main();
