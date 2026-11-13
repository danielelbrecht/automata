"""Default configuration for training and running the NCA.

Any field can be overridden with a JSON file (--config path.json), which may hold
just the keys you want to change, and/or on the command line with
--set section.key=value, e.g.

    python main.py -t --img images/lizard-emoji_40.png --model m.pt \
        --set train.n_iter=2000 --set train.eval_every=250 --set device=cpu
"""
import ast
import json
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    internal_dim: int = 32      # hidden channels per cell, on top of RGB + alpha
    mlp_dim: int = 128          # width of the shared per-cell update network
    pmask: float = 0.5          # probability a cell skips an update (stochastic masking)
    norm: str = 'none'          # 'none' (as in the original paper) or 'layernorm' (the old global LayerNorm)
    world_scale: int = 1       # the world is this many times larger than the image per side (the image
                                # stays the same size and sits in the centre, the rest is empty space)
    bg_tolerance: float = 0.1   # if the image is fully opaque, make the flat background transparent: pixels connected
                                # to the border and within this colour distance of the corner colour (0 = off)
    rgba: bool = True           # paper-style: transparent canvas, seed has alpha + hidden = 1,
                                # premultiplied-RGBA target and loss. False = old white-canvas RGB mode.


@dataclass
class TrainConfig:
    n_iter: int = 20000          # number of optimiser steps
    batch_size: int = 8
    lr: float = 2e-3
    lr_decay_at: int = 2000     # iteration at which lr is multiplied by lr_decay_factor (0 = never)
    lr_decay_factor: float = 0.1
    normalize_grads: bool = True    # divide each parameter's gradient by its norm (as in the paper)
    pool_size: int = 1024       # sample pool of past states (0 = fresh seed every iteration)
    damage_n: int = 3           # per batch, how many of the lowest-loss pool samples get a circle erased
    n_steps: Tuple[int, int] = (64, 120)   # CA steps per rollout, random in [low, high)

    log_every: int = 50         # log a progress line every N iterations
    eval_every: int = 500       # run a full inference and save a GIF every N iterations (0 = off)
    eval_steps: int = 120       # CA steps in each of those inference runs
    gif_dir: Optional[str] = None   # default: <model name>_gifs/ next to the model file
    gif_fps: int = 20
    gif_scale: int = 8          # integer upscaling of the tiny grid in saved GIFs


@dataclass
class InferenceConfig:
    steps: int = 120            # CA steps to run
    display: bool = True        # play the result in a matplotlib window
    fps: int = 20               # playback / GIF speed
    save_gif: Optional[str] = None  # write the rollout to this .gif path
    gif_scale: int = 8


@dataclass
class MnistConfig:
    data_dir: str = 'data'          # MNIST is downloaded here on first use
    n_images: int = 5               # train on only the first N training digits (0 = all 60,000)
    n_per_class: int = 1            # train on only the first N training digits of each class
                                    # (0 = off; takes precedence over n_images)
    classes: List[int] = field(default_factory=lambda: [1, 2, 3])   # digit classes to use (empty list = all 0-9)
    eval_on_train: bool = True      # evaluate on the training digits themselves instead of the MNIST test set
    mask_start: float = 0.1         # fraction of each digit's ink hidden at the start of training...
    mask_end: float = 0.7           # ...growing linearly to this (must be < 1 so some ink stays visible)
    curriculum_iters: int = 6000    # iterations over which the mask grows (0 = always mask_end)
    p_masked: float = 0.25          # per batch sample: probability of starting from a freshly masked digit
                                    # (otherwise the sample continues from its pool state)
    p_single_cell: float = 0.25     # per batch sample: probability of a second stage in the same episode: one ink
                                    # cell of the grown result is transplanted onto an empty canvas and must regrow
                                    # the digit; gradients flow back through both stages
    lr_decay_at: int = 12000        # MNIST-only replacement for train.lr_decay_at (0 = never); keep it after
                                    # curriculum_iters so the curriculum runs at the full learning rate
    eval_digits: int = 0            # digits shown in the evaluation GIFs (0 = all of the evaluation set)
    eval_mask: Optional[float] = None   # mask fraction used by mnist_main.py -e (None = mask_end)
    gif_scale: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    mnist: MnistConfig = field(default_factory=MnistConfig)
    device: Optional[str] = None    # None = auto (cuda, then mps, then cpu)


def _update(obj, values, path=''):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise KeyError('unknown config key: {}{}'.format(path, key))
        current = getattr(obj, key)
        if is_dataclass(current):
            _update(current, value, '{}{}.'.format(path, key))
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)


def _nest(keys, value):
    for key in reversed(keys):
        value = {key: value}
    return value


def _parse(raw):
    lowered = raw.strip().lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('none', 'null'):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def load_config(json_path=None, overrides=()):
    cfg = Config()
    if json_path:
        with open(json_path) as f:
            _update(cfg, json.load(f))
    for item in overrides:
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError("override must look like section.key=value, got '{}'".format(item))
        _update(cfg, _nest(key.strip().split('.'), _parse(raw)))
    return cfg
