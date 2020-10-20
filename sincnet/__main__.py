from sincnet.hparam import parser, get_hparams_object, update_hparams
from sincnet.train import train_network


def main(args):

    hparams = get_hparams_object(args.hparams)
    hparams = update_hparams(args, hparams)

    train_network(hparams)

    # Evaluate model here.


if __name__ == "__main__":
    args = parser()
    main(args)
