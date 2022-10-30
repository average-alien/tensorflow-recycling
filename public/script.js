// import * as tf from '@tensorflow/tfjs'

const STATUS = document.getElementById('status');
const IMAGEUP = document.getElementById('imageUp')
const PREDICT_BUTTON = document.getElementById('predict');
const INPUT_WIDTH = 192;
const INPUT_HEIGHT = 192;
const CLASS_NAMES = ["1 - PET", "2 - HDPE", "3 - PVC", "4 - LDPE", "5 - PP", "6 - PS", "7 - OTHER"];
const IMAGE = document.getElementById("image")
const PREVIEW = document.getElementById("preview")

let model = undefined;
let userImage = undefined
let predict = false;

PREDICT_BUTTON.addEventListener('click', makePrediction);
IMAGEUP.addEventListener('change', e => {
    IMAGE.src = URL.createObjectURL(e.target.files[0])
    PREVIEW.src = URL.createObjectURL(e.target.files[0])
    IMAGE.onload = () => {
        URL.revokeObjectURL(IMAGE.src)
    }
    PREVIEW.onload = () => {
        URL.revokeObjectURL(PREVIEW.src)
    }
  })

function makePrediction() {
    if (predict) {
        tf.tidy(function() {
          userImage = tf.browser.fromPixels(IMAGE).div(1.0)
          let resizedTensor = tf.image.resizeBilinear(userImage, [INPUT_WIDTH, INPUT_HEIGHT], true)
          let prediction = model.predict(resizedTensor.expandDims()).softmax().squeeze()
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
      let answer = model.predict(tf.zeros([1, INPUT_WIDTH, INPUT_HEIGHT, 3]));
      console.log(answer.softmax().squeeze().print());
    });
  }
  
// Call the function immediately to start loading.
loadModel();