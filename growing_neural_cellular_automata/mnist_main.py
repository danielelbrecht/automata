"""Train or evaluate the MNIST NCA (see mnist_nca.py for how training works).

    python mnist_main.py -t --model saved_models/mnist.pt
    python mnist_main.py -e --model saved_models/mnist.pt --set inference.display=false

Evaluation GIF rows: target | grown from the masked digit | regrown from one cell of that result.
"""
import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch

from config import load_config
from main import pick_device
from mnist_nca import demo_grid, load_digits, train_mnist
from nca_model import NeuralCellularAutomata
from viz import play_frames, save_gif

logger = logging.getLogger('mnist')

IMAGE_SIZE = 28


def build_mnist_model(cfg, device):
    if not cfg.model.rgba:
        raise SystemExit('MNIST training needs model.rgba=true')
    if cfg.model.norm != 'none':
        logger.warning('model.norm=%s has per-position parameters, so the model can learn where it is in the grid',
                       cfg.model.norm)
    # The model's own single target is not used for MNIST; a blank image just sets the size.
    return NeuralCellularAutomata(torch.zeros(4, IMAGE_SIZE, IMAGE_SIZE),
                                  internal_dim=cfg.model.internal_dim,
                                  mlp_dim=cfg.model.mlp_dim,
                                  pmask=cfg.model.pmask,
                                  lr=cfg.train.lr,
                                  norm=cfg.model.norm,
                                  rgba=True,
                                  world_scale=cfg.model.world_scale).to(device)


def main():
    parser = argparse.ArgumentParser(description='Grow MNIST digits with a neural cellular automaton')
    parser.add_argument('-t', '--train', help='Do training', action='store_true')
    parser.add_argument('-e', '--eval', help='Do evaluation', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Path to save or load model from')
    parser.add_argument('--config', type=str, help='JSON file overriding any defaults from config.py')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='Override one config value, e.g. --set mnist.mask_end=0.8 (repeatable)')
    args = parser.parse_args()

    if not (args.train or args.eval):
        parser.error('specify at least one of -t/--train or -e/--eval')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    cfg = load_config(args.config, args.overrides)
    logger.info('Config: %s', json.dumps(asdict(cfg)))

    device = pick_device(cfg.device)
    logger.info('Using device: %s', device)

    nca = build_mnist_model(cfg, device)
    mcfg = cfg.mnist
    digits = load_digits(mcfg.data_dir, train=True, n_images=mcfg.n_images, n_per_class=mcfg.n_per_class,
                         classes=mcfg.classes, device=device)
    if mcfg.eval_on_train:
        test_digits = digits
        logger.info('Evaluating on the training digits (mnist.eval_on_train=true)')
    else:
        test_digits = load_digits(mcfg.data_dir, train=False, classes=mcfg.classes, device=device)

    if args.train:
        train_mnist(nca, digits, test_digits, cfg.train, cfg.mnist, args.model)

    if args.eval:
        nca.load_state_dict(torch.load(args.model, map_location=device))
        frac = cfg.mnist.eval_mask if cfg.mnist.eval_mask is not None else cfg.mnist.mask_end
        frames, mse = demo_grid(nca, test_digits, frac, cfg.inference.steps, cfg.mnist.eval_digits)
        logger.info('%d test digits, mask %.2f, %d steps: MSE from masked %.5f, from single cell %.5f',
                    test_digits.shape[0] if mcfg.eval_digits <= 0 else min(mcfg.eval_digits, test_digits.shape[0]), frac, cfg.inference.steps,
                    mse['masked'], mse['cell'])

        model_path = Path(args.model)
        gif_path = cfg.inference.save_gif or str(model_path.with_name(model_path.stem + '_mnist_eval.gif'))
        save_gif(frames, gif_path, fps=cfg.inference.fps, scale=cfg.mnist.gif_scale)
        logger.info('Saved GIF to %s', gif_path)

        if cfg.inference.display:
            play_frames(frames, fps=cfg.inference.fps)


if __name__ == '__main__':
    main()
