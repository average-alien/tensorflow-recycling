# https://www.youtube.com/watch?v=q7ZuZ8ZOErE
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial18-customdata-images/1_in_subfolders.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
keras = tf.keras # had to modify this cause imports were throwing weird errors
layers = tf.keras.layers
# ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator # method 2 (deprecated, it seems method one is prefered)
import numpy as np

print(tf.__version__)

# set parameters of dataset
img_height = 100
img_width = 100
batch_size = 2
model = keras.Sequential(
    [
        layers.Input((100, 100, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

# method 1
# make training set from mnist_subfolder (shuffled data)
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "temp/recycling_symbols/",
    labels="inferred",
    label_mode="int",  # alternatives: categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),    # reshape if not in this size
    shuffle=True,
    seed=123,
    # validation_split=0,    # originally 0.1 (setting aside 10% of data for validation, but I made separate test subfolder)
    # subset="training",
)

# make validation set from test subfolder
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "temp/test/",    # was same as ds_train
    labels="inferred",
    label_mode="int",
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),    # reshape if not in this size
    shuffle=True,
    seed=123,
    # validation_split=1,    # originally 0.1
    # subset="validation",
)

# Compile and Train
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(ds_train, epochs = 10, verbose = 2)

test_loss, test_acc = model.evaluate(ds_validation, verbose=2)
print('\nTest accuracy:', test_acc)    # not sure why it outputs 0 (maybe b/c batch size is 2 and having only 1 file in test subfolder is "odd"; see console...)

# Predict
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(ds_validation)
print(predictions[0])
print(np.argmax(predictions[0]))

# method 2
# [...]