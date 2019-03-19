const model = require('./model.js').setupModel();
const Data = require('./data.js');
const tf = require('@tensorflow/tfjs-node-gpu');
const d = new Data();

const TRAIN_BATCH = 100;

async function main() {
    let trainedModel = await train();
    const resultModel = await trainedModel.save('file://./crepe');
}

async function train() {
  for (let TRAIN_COUNT = 0; TRAIN_COUNT < TRAIN_BATCH; TRAIN_COUNT++) {
    let { xs, label } = d.nextTrainBatch();
    let valid = d.nextValidBatch();

    console.log(`Train[${TRAIN_COUNT}]`);

    const h = await model.fit(xs, label, {
      batchSize: 90,
      epochs: 32,
      shuffle: true,
      validationData: [ valid.xs, valid.label ],
      callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
    });
    console.log("Loss : " + h.history.loss[0]);

    const resultModel = await model.save('file://./crepe');
    xs.dispose();
    label.dispose();
  }

  return model;
}

main();
