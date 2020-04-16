import numpy as np


class EmpiricalNormalizer(object):
    def __init__(self,
                 shape,
                 batch_axis=0,
                 eps=1e-2,
                 dtype=np.float32,
                 until=None,
                 clip_threshold=None):
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self._var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0

        # 1/var
        self._cached_std_inverse = None

    @property
    def mean(self):
        return np.squeeze(self._mean, self.batch_axis)

    @property
    def std(self):
        return np.sqrt(np.squeeze(self._var, self.batch_axis))

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = np.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = np.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        if self.count == 0:
            return x

        mean = np.broadcast_to(self._mean, x.shape)
        std_inv = np.broadcast_to(self._std_inverse, x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = np.clip(
                normalized, -self.clip_threshold, self.clip_threshold
            )

        return normalized













