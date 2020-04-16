from scipy.signal import lfilter


def discount_cumsum(x, discount):
    '''
    :公式 a[0]*y[n] + a[1]*y[n-1] + ... + a[n]*y[0] = b[0]*x[n] + b[1]*x[n-1] + ... + b[n]*x[0]
    :param x: (np.ndarray or tf.Tensor)
    :param discount:
    :return: output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    '''
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0
    )[::-1]    # 数字滤波器 y[n] = x[n] + discount*y[n-1]

