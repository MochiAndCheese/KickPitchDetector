const tf = require('@tensorflow/tfjs-node-gpu');
const SHAPE = 512;

function setupModel() {
  const model = tf.sequential();

  model.add(tf.layers.reshape({
    inputShape: [SHAPE],
    targetShape: [SHAPE, 1, 1]
  }));

  model.add(tf.layers.conv2d({
    kernelSize: 512,
    filters: 128,
    strides: 4,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 1]
  }));
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.conv2d({
    kernelSize: 64,
    filters: 16,
    strides: 1,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 1]
  }));
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.conv2d({
    kernelSize: 64,
    filters: 16,
    strides: 1,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 1]
  }));
  model.add(tf.layers.dropout(0.25));

  model.add(tf.layers.conv2d({
    kernelSize: 64,
    filters: 32,
    strides: 1,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 1]
  }));
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.conv2d({
    kernelSize: 64,
    filters: 64,
    strides: 1,
    padding: 'same',
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 1]
  }));
  model.add(tf.layers.dropout(0.5));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
    units: 24,
    activation: 'softmax'
  }));

  const LEARNING_RATE = 0.0002;
  const optimizer = tf.train.adam(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

module.exports = {
  setupModel: setupModel
}
