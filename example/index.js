const detector = require('../index');
const wavEncoder = require('wav-encoder');

(async () => {
  const wav = await wavEncoder.encode({
    sampleRate: 44100,
    channelData: [
      new Float32Array(44100).map(() => Math.random() - 0.5),
      new Float32Array(44100).map(() => Math.random() - 0.5)
    ]
  });
  console.log(await detector(wav));
})();
