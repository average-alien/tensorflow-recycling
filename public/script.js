// import * as tf from '@tensorflow/tfjs'

const STATUS = document.getElementById('status');
const IMAGEUP = document.getElementById('imageUp')
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 192;
const MOBILE_NET_INPUT_HEIGHT = 192;
const CLASS_NAMES = ["1", "2", "3", "4", "5", "6", "7"];
const IMAGE = document.getElementById("image")

let model = undefined;
let userImage = undefined
let predict = false;

TRAIN_BUTTON.addEventListener('click', makePrediction);
IMAGEUP.addEventListener('change', e => {
    IMAGE.src = URL.createObjectURL(e.target.files[0])
    IMAGE.onload = () => {
        URL.revokeObjectURL(IMAGE.src)
    }
})

function makePrediction() {
    if (predict) {
        tf.tidy(function() {
          userImage = tf.browser.fromPixels(IMAGE).div(127.5).sub(1)
          console.log(userImage.print())
          let prediction = model.predict(userImage.expandDims()).softmax().squeeze()
          console.log(prediction.print())
          let highestIndex = prediction.argMax().arraySync();
          let predictionArray = prediction.arraySync();
    
          STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        });
      }
}

/**
 * Loads the model and warms it up so ready for use.
 **/
async function loadModel() {
    const URL = "http://localhost:8000/model";
    
    model = await tf.loadGraphModel(URL);
    STATUS.innerText = 'Model loaded successfully!';
    predict = true;
    
    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = model.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.softmax().squeeze().print());
    });
  }
  
// Call the function immediately to start loading.
loadModel();