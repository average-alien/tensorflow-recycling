import * as tf from '@tensorflow/tfjs-node';
import { loadGraphModel } from '@tensorflow/tfjs-node';

const MODEL_PATH = "file://js-models/4/model.json"

const model = await loadGraphModel(MODEL_PATH)

const test = tf.browser.fromPixels("https://res.cloudinary.com/dazgyyyvj/image/upload/v1665686727/cfj1utkzwjrmrky8ueye.png")

let result = model.predict(test)

console.log(result)