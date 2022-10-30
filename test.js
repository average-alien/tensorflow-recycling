import * as tf from '@tensorflow/tfjs'
import * as tfn from '@tensorflow/tfjs-node'

const handler = tfn.io.fileSystem('./js-models/4/model.json')

const model = await tf.loadGraphModel(handler)

const prediction = model.execute(tf.zeros([1, 192, 192, 3]))

console.log(prediction.softmax().squeeze().print())