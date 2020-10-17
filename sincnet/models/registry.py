_MODELS = dict()


def register(name):

    def add_to_dict(fn):
        global _MODELS
        _MODELS[name] = fn
        return fn

    return add_to_dict


def get_model(hparams):
    if hparams.model not in _MODELS:
        print('model {} not found'.format(hparams.model))
        exit()
    return _MODELS[hparams.model](hparams)
