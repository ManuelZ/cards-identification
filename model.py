"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# External imports
import tensorflow as tf
from tensorflow.keras.applications import densenet
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Used to investigate exploding calculations
# Doesn't work in TF 2.17 (or maybe when the GPU is used, independently of the TF version. I don't know.)
# tf.debugging.enable_check_numerics()

def cosine_similarity(x, y):
    """ """
    x_norm = tf.nn.l2_normalize(x)
    y_norm = tf.nn.l2_normalize(y)
    return tf.reduce_sum(tf.multiply(x_norm, y_norm))


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
    def __init__(self, siamese_net, loss_fun=None, **kwargs):
        super().__init__()
        self.siamese_net = siamese_net
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_fun = self.triplet_loss if loss_fun == "triplet" else self.circle_loss

    def triplet_loss(self, inputs, margin=0.5):
        """ """

        (anchor_embedding, positive_embedding, negative_embedding) = inputs

        # Square of the Euclidean distance (square of the L2 norm)
        ap_distance = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=-1
        )
        an_distance = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=-1
        )

        loss = ap_distance - an_distance

        loss = tf.maximum(loss + margin, 0.0)

        return loss

    def circle_loss(self, inputs, margin=0.25, scale=256):
        """
        Parameters
            inputs: (anchor embedding, positive embedding, negative embedding)
        """
        q = inputs[0]
        p = inputs[1]
        n = inputs[2]

        # From the paper: "Under the cosine similarity metric, for example, we expect sp->1 and sn->0."
        sim_p = cosine_similarity(q, p)
        sim_n = cosine_similarity(q, n)

        alpha_p = K.relu(-sim_p + 1 + margin)
        alpha_n = K.relu(sim_n + margin)

        margin_p = 1 - margin
        margin_n = margin

        logit_p = -scale * alpha_p * (sim_p - margin_p)
        logit_n = scale * alpha_n * (sim_n - margin_n)

        # Replace the straightforward implementation of the loss that directly uses exp because it can easily explode.
        # 
        # From: 
        # https://github.com/layumi/Person_reID_baseline_pytorch/blob/7497812e72bb859b16152de0863af5a25198dec8/circle_loss.py#L39C9-L39C97
        #
        # Or should it be like this?
        # https://github.com/tensorflow/similarity/blob/dc506d63d05d012d3496041c25c1c54981d39565/tensorflow_similarity/losses/circle_loss.py#L137
        #
        # I think the first form is fine:
        #
        # softplus(x)   = log(exp(x) + 1)
        # softplus(a+b) = log(exp(a+b) + 1)
        # softplus(a+b) = log(1 + exp(a)*exp(b))
        # softplus(logsumexp(logit_n) + logsumexp(logit_p)) = log(1 + exp(logsumexp(logit_n))*exp(logsumexp(logit_p)))
        # softplus(logsumexp(logit_n) + logsumexp(logit_p)) = log(1 + sumexp(logit_n)*sumexp(logit_p))
        #                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ that looks just like in the paper
        loss = tf.math.softplus(tf.math.reduce_logsumexp(logit_n) + tf.math.reduce_logsumexp(logit_p))

        return loss

    def get_config(self):
        """
        See:
        https://www.tensorflow.org/guide/keras/serialization_and_saving#custom_objects
        https://www.tensorflow.org/guide/keras/serialization_and_saving#config_methods
        """
        base_config = super().get_config()
        config = {"siamese_net": self.siamese_net}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["siamese_net"] = tf.keras.layers.deserialize(config["siamese_net"])
        return cls(**config)

    def train_step(self, inputs):
        """
        inputs: (anchor, positive, negative)
        """

        with tf.GradientTape() as tape:
            embeddings = self.siamese_net(inputs)
            loss = self.loss_fun(embeddings)

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
        embeddings = self.siamese_net(inputs)
        loss = self.loss_fun(embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        """
        The model will call reset_states() on any object listed here at the beginning of
        each fit() epoch or at the beginning of a call to evaluate().
        """
        return [self.loss_tracker]

    # def call(self, inputs):
    #     """

    #     Parameters
    #         inputs: (anchor, positive, negative)
    #     """
    #     return self._compute_distance(inputs)
