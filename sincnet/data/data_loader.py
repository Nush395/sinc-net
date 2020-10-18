import tensorflow as tf
import tensorflow_datasets as tfds

AUDIO_SHAPE = (16000,)


def data_loader(hparams, splits):
    """
        only train and valid for now.
    """
    ds_tuples, info = tfds.load(name=hparams.dataset,
                                data_dir=hparams.data_dir,
                                split=splits,
                                with_info=True,
                                download=True,
                                as_supervised=True)
    ds_datasets = []
    for ds, split_name in zip(ds_tuples, splits):
        if split_name == "train":
            ds = ds.map(lambda x, y: (tf.cast(x, dtype=tf.float32)/32767.0, y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # apply augs here.

            ds = ds.shuffle(1000)
            ds = ds.padded_batch(hparams.batch_size,
                                 padded_shapes=(AUDIO_SHAPE, ()))

            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            ds = ds.padded_batch(hparams.batch_size,
                                 padded_shapes=(AUDIO_SHAPE, ()),
                                 padding_values=tf.constant(0, dtype=tf.int64))
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        ds_datasets.append(ds)

    return ds_datasets, info
