import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/4")

# file paths
data_dir = "data/recycling_symbols_filtered"
test_dir = "data/test"

# ds settings
batch_size = 64
img_height = 192
img_width = 192

# loading images into datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    # validation_split=0.2, 
    # subset="training",
    # seed=39,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     labels='inferred',
#     label_mode='int',
#     validation_split=0.2, 
#     subset="validation",
#     seed=39,
#     color_mode='rgb',
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    shuffle=False,
    color_mode='rgb',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = ["1", "2", "3", "4", "5", "6", "7"]

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(train_ds)

for images, labels in train_ds.take(1):
    for i in range(7):
        print("-------------------------------------")
        print(predictions[i])
        print("expected:", class_names[labels[i]])
        print("prediction:", class_names[np.argmax(predictions[i])])
    break