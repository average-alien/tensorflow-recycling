# https://www.tensorflow.org/tutorials/keras/classification
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(np.sin((np.linspace(-np.pi, np.pi, 10001))))
# plt.show()

# Import fashion mnist
# fashion_mnist = tf.keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
batch_size = 32
img_height = 128
img_width = 128
train_images = tf.keras.utils.image_dataset_from_directory(
    "data/recycling_symbols/",
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_images = tf.keras.utils.image_dataset_from_directory(
    "data/recycling_symbols/",
    labels='inferred',
    label_mode='int',
    validation_split=0.1, 
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = ["Recyc1", "Recyc2", "Recyc3", "Recyc4", "Recyc5", "Recyc6", "Recyc7"]

# for images, labels in test_images.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()

# Preprocess data: 0=black 255=white -> 0=black 1=white
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# normalization_layer = tf.keras.layers.Rescaling(1./255)
# train_images = train_images.map(lambda x, y: (normalization_layer(x), y))
# test_images = train_images.map(lambda x, y: (normalization_layer(x), y))

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train the model
model.fit(train_images, epochs = 50)

test_loss, test_acc = model.evaluate(test_images, verbose=2)
print('\nTest accuracy:', test_acc)

# img = tf.keras.preprocessing.image.load_img(
#     "data/recycling-test/7/IMG_6312.jpg",
#     target_size=(img_height, img_width)
# )
# img_array = tf.keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)
# predictions = model.predict(img_array)
# score = predictions[0]
# print(score)


# Make PREDICITONS
# "Attach a softmax layer to convert the model's linear outputs—logits—to probabilities"
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# since test_images has 10k images, predictions will be an array containing 10k arrays, each having 10 values representing the probabilities that an item is one of the 10 categories
predictions = probability_model.predict(test_images)
# return the highest confidence value
# np.argmax(predictions[0])
# print(np.argmax(predictions[0]))
# ^ should return "9", meaning the first image in test_images is predicted to be in the category 9 => Ankle boot (which is correct)

for images, labels in test_images.take(1):
    for i in range(9):
        print("-------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:",class_names[np.argmax(predictions[i])])
    break