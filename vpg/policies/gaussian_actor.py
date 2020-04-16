import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from distribution.diagonal_gaussian import DiagonalGaussian


class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-20 = 2.061e-9

    def __init__(self,
                 state_shape,
                 action_dim,
                 max_action,
                 units=[256, 256],
                 hidden_activation="relu",
                 fix_std=False,
                 const_std=0.1,
                 state_independent_std=False,
                 name='GaussianPolicy'):
        super(GaussianActor, self).__init__(name=name)
        self.dist = DiagonalGaussian(dim=action_dim)
        self._fix_std = fix_std
        self._const_std = const_std
        self._max_action = max_action
        self._state_independent_std = state_independent_std

        self.l1 = Dense(units[0], name="L1", activation=hidden_activation)
        self.l2 = Dense(units[1], name="L2", activation=hidden_activation)
        self.out_mean = Dense(action_dim, name="L_mean")

        if not self._fix_std:
            # 判断是否独立分布
            if self._state_independent_std:
                self.out_log_std = tf.Variable(
                    initial_value=0.5*np.ones(action_dim, dtype=np.float32),
                    dtype=tf.float32, name="logstd"
                )
            else:
                self.out_log_std = Dense(action_dim, name="L_sigma")

        self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

    def _compute_dist(self, states):
        features = self.l1(states)
        features = self.l2(features)
        mean = self.out_mean(features)

        if self._fix_std:
            log_std = tf.ones_like(mean) * tf.math.log(self._const_std)
        else:
            if self._state_independent_std:
                log_std = tf.tile(
                    input=tf.expand_dims(self.out_log_std, axis=0),
                    multiples=[mean.shape[0], 1]
                )
            else:
                log_std = self.out_log_std(features)
                log_std = tf.clip_by_value(log_std, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        return {"mean": mean, "log_std": log_std}

    def call(self, states, test=False):
        param = self._compute_dist(states)
        if test:
            raw_actions = param["mean"]
        else:
            raw_actions = self.dist.sample(param)
        logp_pis = self.dist.log_likelihood(raw_actions, param)
        actions = raw_actions

        return actions * self._max_action, logp_pis, param

    def compute_log_probs(self, states, actions):
        actions /= self._max_action
        param = self._compute_dist(states)
        logp_pis = self.dist.log_likelihood(actions, param)

        return logp_pis

    def compute_entropy(self, states):
        param = self._compute_dist(states)
        return self.dist.entropy(param)



