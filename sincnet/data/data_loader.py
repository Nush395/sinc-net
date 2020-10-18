import tensorflow as tf
import tensorflow_datasets as tfds


def data_loader(hparams, splits):
    """
        only train and valid for now.
    """
    (train, validation), info = tfds.load(name=hparams.dataset,
                                          data_dir=hparams.data_dir,
                                          split=splits,
                                          with_info=True,
                                          download=True,
                                          as_supervised=True)
    train = train.map(lambda x, y: (tf.cast(x, dtype=tf.float32)/32767.0, y),
    # apply augs here.

    train.shuffle(1000)
    train = train.padded_batch(hparams.batch_size,
                               padded_shapes=((16000, ), ()))

                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    validation = validation.padded_batch(
                    hparams.batch_size,
                    padded_shapes=((16000, ), ()),
                    padding_values=tf.constant(0, dtype=tf.int64))
    validation = validation.prefetch(tf.data.experimental.AUTOTUNE)

    return (train, validation), info
