"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow import keras


def get_embedding_module(image_size):
    """ """

    inputs = keras.Input(image_size + (3,))

    baseCnn = resnet.ResNet50(weights="imagenet", include_top=False)
    baseCnn.trainable = False

    x = resnet.preprocess_input(inputs)
    x = baseCnn(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(units=128)(x)
    embedding = keras.Model(inputs, outputs, name="embedding")

    return embedding


def get_siamese_network(image_size, embedding_model):
    """ """

    anchor_input = keras.Input(name="anchor", shape=image_size + (3,))
    positive_input = keras.Input(name="positive", shape=image_size + (3,))
    negative_input = keras.Input(name="negative", shape=image_size + (3,))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    siamese_network = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding],
    )
    return siamese_network


class SiameseModel(keras.Model):
    def __init__(self, siamese_net, margin, lossTracker):
        super().__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.lossTracker = lossTracker

    def _compute_distance(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        (anchor_embedding, positive_embedding, negative_embedding) = self.siamese_net(
            inputs
        )

        apDistance = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=-1
        )

        anDistance = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=-1
        )

        return (apDistance, anDistance)

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def call(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """
        return self._compute_distance(inputs)

    def train_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        with tf.GradientTape() as tape:
            (apDistance, anDistance) = self._compute_distance(inputs)
            loss = self._compute_loss(apDistance, anDistance)

        gradients = tape.gradient(loss, self.siamese_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_net.trainable_variables)
        )
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    def test_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        (apDistance, anDistance) = self._compute_distance(inputs)
        loss = self._compute_loss(apDistance, anDistance)
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    @property
    def metrics(self):
        return [self.lossTracker]
