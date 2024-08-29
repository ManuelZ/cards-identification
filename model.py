"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
import tensorflow as tf
from tensorflow.keras.applications import resnet, densenet
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def get_embedding_module(image_size):
    """ """

    inputs = tf.keras.Input(image_size + (3,))

    baseCnn = densenet.DenseNet121(weights="imagenet", include_top=False)
    baseCnn.trainable = False

    x = densenet.preprocess_input(inputs)
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
    embedding = tf.keras.Model(inputs, outputs, name="embedding")

    return embedding


def get_siamese_network(image_size, embedding_model):
    """ """

    anchor_input = tf.keras.Input(name="anchor", shape=image_size + (3,))
    positive_input = tf.keras.Input(name="positive", shape=image_size + (3,))
    negative_input = tf.keras.Input(name="negative", shape=image_size + (3,))

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    siamese_network = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding],
    )
    return siamese_network


class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_net, margin, **kwargs):
        super().__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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

    def get_config(self):
        """
        See:
        https://www.tensorflow.org/guide/keras/serialization_and_saving#custom_objects
        https://www.tensorflow.org/guide/keras/serialization_and_saving#config_methods
        """
        base_config = super().get_config()
        config = {
            "siamese_net": self.siamese_net,
            "margin": self.margin,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["siamese_net"] = tf.keras.layers.deserialize(config["siamese_net"])
        return cls(**config)

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def call(self, inputs):
        """

        Parameters
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

        # Compute gradients
        trainable_vars = self.siamese_net.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        (apDistance, anDistance) = self._compute_distance(inputs)
        loss = self._compute_loss(apDistance, anDistance)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        """
        The model will call reset_states() on any object listed here at the beginning of
        each fit() epoch or at the beginning of a call to evaluate().
        """
        return [self.loss_tracker]


class CircleLoss(Model):
    """Modified from: https://github.com/xiangli13/circle-loss/blob/master/circle_loss.py"""

    def __init__(self, scale=256, margin=0.25, similarity="cos", **kwargs):
        self.scale = scale
        self.margin = margin
        self.similarity = similarity
        super().__init__(dynamic=True, **kwargs)

    def call(self, inputs):
        q = inputs[0]
        p = inputs[1]
        n = inputs[2]

        if self.similarity == "dot":
            sim_p = self.dot_similarity(q, p)
            sim_n = self.dot_similarity(q, n)
        elif self.similarity == "cos":
            sim_p = self.cosine_similarity(q, p)
            sim_n = self.cosine_similarity(q, n)
        else:
            raise ValueError("This similarity is not implemented.")

        alpha_p = K.relu(-sim_p + 1 + self.margin)
        alpha_n = K.relu(sim_n + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin

        loss_p = K.sum(K.exp(-self.scale * alpha_p * (sim_p - margin_p)), axis=1)
        loss_n = K.sum(K.exp(self.scale * alpha_n * (sim_n - margin_n)), axis=1)
        return K.log(1 + loss_p * loss_n)

    def compute_output_shape(self, input_shape):
        return (1,)

    def dot_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        return K.dot(x, K.transpose(y))

    def cosine_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        abs_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        abs_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        up = K.dot(x, K.transpose(y))
        down = K.dot(abs_x, K.transpose(abs_y))
        return up / down
