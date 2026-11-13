import argparse
import json
import logging
from dataclasses import asdict

import torch
import torch.nn.functional as tnnf
from torchvision.io import read_image, ImageReadMode

# Local imports
from config import load_config
from nca_model import NeuralCellularAutomata
from viz import play_frames, save_gif

logger = logging.getLogger('nca')


def remove_flat_background(img, tol):
    """Make the flat background of a fully opaque (4, H, W) image transparent.

    A pixel becomes transparent if it is within `tol` (max per-channel difference) of the top-left corner
    colour and is connected to the image border through such pixels, so enclosed areas of the same colour
    (an eye highlight, say) are kept. Images that already have transparency are returned unchanged.
    Returns (image, number of pixels made transparent).
    """
    if tol <= 0 or img.shape[0] != 4 or bool((img[3] < 1.0).any()):
        return img, 0

    rgb = img[:3]
    near = (rgb - rgb[:, 0, 0].view(3, 1, 1)).abs().amax(dim=0) <= tol

    border = torch.zeros_like(near)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    # Flood fill from the border, one pixel ring at a time, restricted to background-coloured pixels
    reached = near & border
    while True:
        grown = tnnf.max_pool2d(reached[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] > 0
        grown = grown & near
        if int(grown.sum()) == int(reached.sum()):
            break
        reached = grown

    out = img.clone()
    out[3][reached] = 0.0
    return out, int(reached.sum())


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(cfg, img_path, device):
    """Load the target image (removing a flat background if needed) and build the model from the config."""
    img = read_image(img_path, ImageReadMode.RGB_ALPHA).to(torch.float32)
    img = img / 255.

    if cfg.model.rgba:
        img, n_removed = remove_flat_background(img, cfg.model.bg_tolerance)
        if n_removed:
            logger.info('Image is fully opaque: made the %d of %d background pixels connected to the border transparent '
                        '(model.bg_tolerance=%g)', n_removed, img.shape[1] * img.shape[2], cfg.model.bg_tolerance)

    return NeuralCellularAutomata(img,
                                  internal_dim=cfg.model.internal_dim,
                                  mlp_dim=cfg.model.mlp_dim,
                                  pmask=cfg.model.pmask,
                                  lr=cfg.train.lr,
                                  norm=cfg.model.norm,
                                  rgba=cfg.model.rgba,
                                  world_scale=cfg.model.world_scale).to(device)


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or run a growing neural cellular automaton')
    parser.add_argument('-t', '--train', help='Do training', action='store_true')
    parser.add_argument('-e', '--eval', help='Do evaluation', action='store_true')
    parser.add_argument('--model', type=str, help='Path to save or load model from')
    parser.add_argument('--img', type=str, help='Path to desired image for training')
    parser.add_argument('--config', type=str, help='JSON file overriding any defaults from config.py')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='Override one config value, e.g. --set train.n_iter=2000 (repeatable)')

    args = parser.parse_args()

    if not (args.train or args.eval):
        parser.error('specify at least one of -t/--train or -e/--eval')
    if args.model is None or args.img is None:
        parser.error('--model and --img are both required')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    cfg = load_config(args.config, args.overrides)
    logger.info('Config: %s', json.dumps(asdict(cfg)))

    device = pick_device(cfg.device)
    logger.info('Using device: %s', device)

    nca = build_model(cfg, args.img, device)

    if args.train:
        nca.train_ca(cfg.train, path=args.model)

    if args.eval:
        icfg = cfg.inference

        # Load trained model and roll the whole sequence out up front: stepping the model is fast,
        # drawing is the slow part, so the frames are only drawn once they all exist.
        nca.load_state_dict(torch.load(args.model, map_location=device))
        frames = nca.rollout(icfg.steps)
        logger.info('Ran %d inference steps', icfg.steps)

        if icfg.save_gif:
            save_gif(frames, icfg.save_gif, fps=icfg.fps, scale=icfg.gif_scale)
            logger.info('Saved GIF to %s', icfg.save_gif)

        if icfg.display:
            play_frames(frames, fps=icfg.fps)
        elif not icfg.save_gif:
            logger.warning('Nothing to do with the result: set inference.display=true or inference.save_gif=PATH')


if __name__ == '__main__':
    main()
