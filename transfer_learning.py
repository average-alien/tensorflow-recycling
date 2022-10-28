import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# file paths
data_dir = "data/recycling_symbols"
test_dir = "data/test"

# ds settings
batch_size = 32
img_height = 192
img_width = 192

# loading images into datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2, 
    subset="training",
    seed=39,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2, 
    subset="validation",
    seed=39,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# pull out the class labels
class_names = train_ds.class_names
print(class_names)

# options for optimizing performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# defining the model layers
num_classes = len(class_names)

# data augmentaion layers
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.3),
  tf.keras.layers.RandomContrast(0.2),
  tf.keras.layers.RandomZoom(0.1),
])

# function to process input for mobilenet
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# base mobilenet model pretrained with imagenet ds
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# freeze the weights
base_model.trainable = False

# forgot what this does
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# final layer in model
prediction_layer = tf.keras.layers.Dense(num_classes)

# define input tensor and start setting up our model
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
# not too sure why its set up with this method
x = data_augmentation(inputs) 
x = preprocess_input(x)

x = base_model(x, training=False) # make sure training is set to false for base model
# x = tf.keras.layers.Dropout(0.1)(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# lower learning rate for transfer learning is better
base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# training the model
epochs = 500

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

# Saving the trained model
saved_model = "models/1" # increment number for new versions

tf.keras.models.save_model(
    model=model,
    filepath=saved_model
)

# Test model against validation ds
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print('\nTest accuracy:', test_acc)

# Predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)

for images, labels in test_ds.take(1):
    for i in range(7):
        print("-------------------------------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
    break

# Plotting training graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# Making sure everything is cleaned up (might not be neccessary)
tf.keras.backend.clear_session()
plt.show()