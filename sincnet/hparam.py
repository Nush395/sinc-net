import argparse


_HPARAMS = dict()


def parse_bool(v):
    """For correct parsing of boolean arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def register(fn):
    global _HPARAMS
    _HPARAMS[fn.__name__] = fn()
    return fn


def get_hparams_object(hparams_str):
    return _HPARAMS[hparams_str]


def update_hparams(FLAGS, hparams):
    # set hparams from FLAGS attribtues
    for attr in vars(FLAGS):
        if attr not in ["h", "help", "helpshort", "env"]:
            if getattr(FLAGS, attr) is not None:
                setattr(hparams, attr, getattr(FLAGS, attr))
    return hparams


class Hyperparameters:
    def __init__(self) -> None:
        self.num_classes = 10
        self.num_epochs = 200
        self.batch_size = 128
        self.lr = 0.1
        self.model = "test_nn"
        self.dataset = "speech_commands"


@register
def baseline():
    hps = Hyperparameters()
    return hps


def parser():
    parser = argparse.ArgumentParser(
        description='Runs experiments on SincNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lr", default=None, type=float, help="learning rate")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()
    return args
