import tensorflow as tf
from distribution.base import Distribution


class Categorical(Distribution):
    def kl(self, old_param, new_param):
        old_prob, new_prob = old_param["prob"], new_param["prob"]

        return tf.reduce_sum(
            old_prob * (tf.math.log(old_prob * self._tiny) - tf.math.log(new_param * self._tiny))
        )

    def likelihood_ratio(self, x, old_param, new_param):
        old_prob, new_prob = old_param["prob"], new_param["prob"]
        return(tf.reduce_sum(new_prob * x, axis=1) + self._tiny) / (tf.reduce_sum(old_prob * x, axis=1) + self._tiny)

    def log_likelihood(self, x, param):
        probs = param["prob"]
        assert probs.shape == x.shape

        return tf.math.log(tf.reduce_sum(probs * x, axis=1) + self._tiny)

    def sample(self, param):
        probs = param["prob"]
        return tf.random.categorical(tf.math.log(probs), 1)

    def entropy(self, param):
        probs = param["prob"]
        return -tf.reduce_sum(probs * tf.math.log(probs + self._tiny), axis=1)

