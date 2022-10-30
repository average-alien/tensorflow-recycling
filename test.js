import * as tf from '@tensorflow/tfjs-node'

const model = await tf.loadGraphModel('file://js-models/4/model.json')

const prediction = model.predict(tf.zeros([1, 192, 192, 3]))

console.log(prediction.softmax().squeeze().print())