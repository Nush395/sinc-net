import tensorflow as tf

from sincnet.models import get_model
from sincnet.data import data_loader


def train_network(hparams):

    (train, validation), info = data_loader(hparams,
                                            splits=['train', 'validation'])
    train_size = info.splits['train'].num_examples
    print("Number of samples: ", train_size)
    model = get_model(hparams)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=loss,
                  metrics=['accuracy'])
    model.summary()
    model.fit(train, validation_data=validation, epochs=hparams.num_epochs,
              verbose=1)
