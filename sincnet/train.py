import tensorflow_datasets as tfds


from sincnet.models import get_model


def train_network(hparams):

    (train, validation), info = tfds.load(name=hparams.dataset,
                                          data_dir=hparams.data_dir,
                                          split=["train", "validation"],
                                          with_info=True,
                                          download=True)
    train_size = info.splits['train'].num_examples
    print(train_size)

    import ipdb; ipdb.set_trace()
    model = get_model(hparams)
    pass
