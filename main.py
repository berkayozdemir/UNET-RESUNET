import tensorflow as tf
import keras.optimizers
import glob
import cv2

import numpy as np
from matplotlib import pyplot as plt

from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
print(device_lib.list_local_devices())
import os
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


IMG_H = 512
IMG_W = 512
NUM_CLASSES = 5
global CLASSES
global COLORMAP

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Load and split the dataset """
def     load_dataset(path, split=0.1):
    train_x = sorted(glob.glob("train/train_val/*.png"))
    train_y = sorted(glob.glob("train/mask_val/*.png"))

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def get_colormap(path):

    x = np.array([[255, 255 ,255],[255 ,0 ,0],[0 ,255 ,0],[0 ,0 ,255],[255 ,0 ,255]], np.uint8)
    colormap = x

    colormap = [[c[0], c[1], c[2]] for c in colormap]

    classes = [
        "Background",
        "Image",
        "Text",
        "FormElement",
        "Container"
    ]

    return classes, colormap

def read_image_mask(x, y):
    """ Reading """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_COLOR)
    assert x.shape == y.shape

    """ Resizing """
  #  x = cv2.resize(x, (IMG_W, IMG_H))
   # y = cv2.resize(y, (IMG_W, IMG_H))

    """ Image processing """
    x = x / 255.0
    x = x.astype(np.float32)

    """ Mask processing """
    output = []
    for color in COLORMAP:
        cmap = np.all(np.equal(y, color), axis=-1)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)


    return x, output

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        image, mask = read_image_mask(x, y)

        return image, mask


    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])
    
    return image, mask


def tf_dataset(x, y, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")


    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 4
    num_epochs = 100
    dataset_path = ""
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Process the colormap """
    CLASSES, COLORMAP = get_colormap(dataset_path)

    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)


from keras.models import Model
from keras.layers import Input, Conv2D,  Conv2DTranspose, BatchNormalization

from keras.layers import Activation, MaxPool2D, Concatenate



from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1e-7):
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    y_pred_ = y_pred.astype(np.float32)
    intersection = K.sum(y_true * y_pred_, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred_, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)



def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



import unet
#model = unet.unet()
model = unet.resunet(IMG_H,NUM_CLASSES)
optimizer = keras.optimizers.Adam(learning_rate=0.00005)
metrics = [dice_coef, 'accuracy']
model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=metrics)
model.summary()

callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path, append=True),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=num_epochs,
          callbacks=callbacks
          )

model.save('multiclass.hdf5')
history = model.history()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show(block=False)



