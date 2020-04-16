import numpy as np
import tensorflow as tf

from distribution.base import Distribution


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_param, new_param):
        old_means, old_log_stds = old_param["mean"], old_param["log_std"]
        new_means, new_log_stds = new_param["mean"], new_param["log_std"]
        old_std = tf.math.exp(old_log_stds)  # 去对数化
        new_std = tf.math.exp(new_log_stds)

        numerator = tf.math.square(old_means - new_means) + tf.math.square(old_std) - tf.math.square(new_std)
        denominator = 2 * tf.math.square(new_std)
        kl = tf.math.reduce_sum(numerator / denominator + new_log_stds - old_log_stds)
        return kl

    def likelihood_ratio(self, x, old_param, new_param):
        llh_new = self.log_likelihood(new_param)
        llh_old = self.log_likelihood(old_param)
        return tf.math.exp(llh_new - llh_old)

    def log_likelihood(self, x, param):
        """"
        :param x: actions
        :param param:
            means: (batch_size, output_dim)
            log_std: (batch_size, output_dim)
        :return: log π(a|s)
        """
        means = param["mean"]
        log_stds = param["log_std"]
        assert means.shape == log_stds.shape
        zs = (x - means) / tf.math.exp(log_stds)  # zs = [actions - u(s)] / σ
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * tf.math.log(2 * np.pi)

    def sample(self, param):
        means = param["mean"]
        log_stds = param["log_std"]
        return means + tf.random.normal(shape=means.shape) * tf.math.exp(log_stds)  # actions = u(s) + z * σ

    def entropy(self, param):
        pass




