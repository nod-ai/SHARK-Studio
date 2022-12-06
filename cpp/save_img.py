import numpy as np
import tensorflow as tf
from shark.shark_inference import SharkInference


def load_and_preprocess_image(fname: str):
    image = tf.io.read_file(fname)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    # preprocessing pipeline
    input_tensor = tf.keras.applications.resnet50.preprocess_input(image)
    return input_tensor


data = load_and_preprocess_image("dog_imagenet.jpg").numpy()

data.tofile("dog.bin")
