import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

@tf.function
def minus_if_zero(a, b):
    c = tf.math.subtract(a,b)
    c = c * b
    c = tf.math.divide_no_nan(c, b)
    return c

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    a = tf.convert_to_tensor([[[[1, 2, 3],[4, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]]],[[[1, 2, 3],[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3],[1, 2, 3]]]],dtype=tf.float32)
    print(a.shape)
    b = tf.convert_to_tensor([[[[0, 0, 0],[0.5, 0, 0],[0, 0, 0]],[[0, 0, 0],[0, 0, 0],[0, 0, 0]],[[0, 0, 0],[0, 0, 0],[0, 0, 0]]],[[[0, 0, 0],[0, 0, 0],[0, 0, 0]],[[0, 0, 0],[0, 0, 0],[0, 0, 0]],[[0, 0, 0],[0, 0, 0],[0, 0, 0]]]])
    # b = tf.convert_to_tensor([[1, 2, 3],[3, 2, 1]])
    print(b.shape)
    # c = tf.einsum('ijkl,ijkl->ijkl', a, b).numpy()
    c = minus_if_zero(a,b)
    cn = c.numpy()
    print(c.shape)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
