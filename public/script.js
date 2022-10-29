const STATUS = document.getElementById('status');
const IMAGEUP = document.getElementById('imageUp')
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 192;
const MOBILE_NET_INPUT_HEIGHT = 192;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = ["1", "2", "3", "4", "5", "6", "7"];
const IMAGE = document.getElementById("image")

let model = undefined;
let userImage = undefined
let predict = false;

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', makePrediction);
RESET_BUTTON.addEventListener('click', reset);
IMAGEUP.addEventListener('change', e => {
    IMAGE.src = URL.createObjectURL(e.target.files[0])
    IMAGE.onload = () => {
        URL.revokeObjectURL(IMAGE.src)
    }
})

function enableCam() {

}


function makePrediction() {
    if (predict) {
        tf.tidy(function() {
          let videoFrameAsTensor = tf.browser.fromPixels(IMAGE).div(255);
          let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT, 
              MOBILE_NET_INPUT_WIDTH], true);
        let reshapedTensor = tf.reshape(resizedTensorFrame, [-1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    
          let prediction = model.predict(reshapedTensor).squeeze();
          let highestIndex = prediction.argMax().arraySync();
          let predictionArray = prediction.arraySync();
    
          STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        });
      }
}


function reset() {
  // TODO: Fill this out later in the codelab!
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
      console.log(answer.shape);
    });
  }
  
// Call the function immediately to start loading.
loadModel();