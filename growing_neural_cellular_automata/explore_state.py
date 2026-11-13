"""Interactive explorer for the internal state of a trained NCA.

Loads a model, runs it from the seed for N steps, then shows two panels:
  left  - the generated image; click any cell to select it
  right - how similar every other cell's state vector is to the selected cell's, colour coded

Cells that are not alive (reset to the blank canvas by the live-cell mask) have no state and are drawn grey.

Keys:  m = cycle the similarity measure (cosine / correlation / euclidean)
       n = switch between comparing the hidden channels only and all channels (RGBA + hidden)

Example:
    python explore_state.py --model saved_models/lizard.pt --img images/lizard-emoji_40.png --steps 100 --seed 0
"""
import argparse
import logging

import numpy as np
import torch

from config import load_config
from main import build_model, pick_device

METRICS = ('cosine', 'correlation', 'euclidean')
CHANNEL_SETS = ('hidden', 'all')


def compute_similarity(feats, alive, y, x, metric):
    """feats: (F, H, W) state vectors. Returns an (H, W) map of how close each cell is to cell (y, x).

    cosine / correlation are in [-1, 1] (1 = same direction); euclidean is a distance (0 = identical).
    Cells that are not alive are NaN.
    """
    n_feat, height, width = feats.shape
    if not alive[y, x]:
        return np.full((height, width), np.nan)

    f = feats.reshape(n_feat, -1).T.astype(np.float64)  # (cells, features)
    v = f[y * width + x]

    if metric == 'correlation':
        # Pearson correlation across the feature vector = cosine similarity after removing each vector's mean
        f = f - f.mean(axis=1, keepdims=True)
        v = v - v.mean()

    if metric in ('cosine', 'correlation'):
        denom = np.linalg.norm(f, axis=1) * np.linalg.norm(v)
        sim = (f @ v) / np.maximum(denom, 1e-12)
    else:
        sim = np.linalg.norm(f - v, axis=1)

    sim = sim.reshape(height, width)
    sim[~alive] = np.nan
    return sim


class Explorer:

    def __init__(self, state, rgb, alive, steps, channels='hidden', metric='cosine', cell=None):
        import matplotlib.pyplot as plt
        self.plt = plt

        self.state = state    # (C, H, W) numpy
        self.alive = alive    # (H, W) bool
        self.channels = channels
        self.metric = metric
        self.steps = steps
        self.height, self.width = alive.shape
        self.y, self.x = cell

        self.fig, (self.ax_img, self.ax_sim) = plt.subplots(1, 2, figsize=(13, 6.5))

        self.ax_img.imshow(rgb, interpolation='nearest')
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        self.sim_im = self.ax_sim.imshow(np.full((self.height, self.width), np.nan), interpolation='nearest')
        self.ax_sim.set_xticks([])
        self.ax_sim.set_yticks([])
        self.cbar = self.fig.colorbar(self.sim_im, ax=self.ax_sim, fraction=0.046, pad=0.04)

        marker = dict(marker='o', markerfacecolor='none', markeredgecolor='red', markersize=12, markeredgewidth=2)
        self.marker_img, = self.ax_img.plot([self.x], [self.y], linestyle='none', **marker)
        self.marker_sim, = self.ax_sim.plot([self.x], [self.y], linestyle='none', **marker)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update()

    def update(self):
        plt = self.plt
        feats = self.state[4:] if self.channels == 'hidden' else self.state
        sim = compute_similarity(feats, self.alive, self.y, self.x, self.metric)

        if self.metric == 'euclidean':
            cmap_name, vmin, label = 'viridis_r', 0.0, 'Euclidean distance (smaller = more similar)'
            vmax = float(np.nanmax(sim)) if np.isfinite(sim).any() else 1.0
            vmax = max(vmax, 1e-6)
        else:
            cmap_name, vmin, vmax = 'coolwarm', -1.0, 1.0
            label = '{} similarity'.format(self.metric)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad('0.88')

        self.sim_im.set_data(sim)
        self.sim_im.set_cmap(cmap)
        self.sim_im.set_clim(vmin, vmax)
        self.cbar.set_label(label)

        for marker in (self.marker_img, self.marker_sim):
            marker.set_data([self.x], [self.y])

        alpha = self.state[3, self.y, self.x]
        if self.alive[self.y, self.x]:
            hidden_norm = float(np.linalg.norm(self.state[4:, self.y, self.x]))
            info = 'cell (y={}, x={})   alpha {:.2f}   |hidden| {:.2f}'.format(self.y, self.x, alpha, hidden_norm)
        else:
            info = 'cell (y={}, x={})   empty (not alive, no state)'.format(self.y, self.x)
        self.ax_img.set_title('Generated image after {} steps\n{}'.format(self.steps, info), fontsize=10)
        self.ax_sim.set_title('{} of {} channels vs selected cell\n[m] measure  [n] channels'.format(
            self.metric, self.channels), fontsize=10)

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.button != 1 or event.inaxes not in (self.ax_img, self.ax_sim):
            return
        toolbar = self.fig.canvas.toolbar
        if toolbar is not None and toolbar.mode != '':
            return  # zoom / pan tool is active
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < self.width and 0 <= y < self.height:
            self.x, self.y = x, y
            self.update()

    def on_key(self, event):
        if event.key == 'm':
            self.metric = METRICS[(METRICS.index(self.metric) + 1) % len(METRICS)]
            self.update()
        elif event.key == 'n':
            self.channels = CHANNEL_SETS[(CHANNEL_SETS.index(self.channels) + 1) % len(CHANNEL_SETS)]
            self.update()


def main():
    parser = argparse.ArgumentParser(description='Explore the internal state of cells in a trained NCA')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--img', type=str, required=True, help='Path to the image the model was trained on')
    parser.add_argument('--config', type=str, help='JSON file overriding any defaults from config.py')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='Override one config value (repeatable), e.g. --set model.world_scale=2')
    parser.add_argument('--steps', type=int, help='Inference steps before capturing the state (default: inference.steps)')
    parser.add_argument('--seed', type=int, help='Random seed, for a repeatable run (the update mask is stochastic)')
    parser.add_argument('--channels', choices=CHANNEL_SETS, default='hidden',
                        help="Compare the hidden channels only, or all channels incl. RGBA (default: hidden)")
    parser.add_argument('--metric', choices=METRICS, default='cosine', help='Similarity measure (default: cosine)')
    parser.add_argument('--cell', type=int, nargs=2, metavar=('Y', 'X'), help='Initially selected cell (default: the seed cell)')
    parser.add_argument('--save', type=str, help='Save the dashboard to this image file and exit instead of opening a window')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger('explore')

    if args.save:
        import matplotlib
        matplotlib.use('Agg')

    cfg = load_config(args.config, args.overrides)
    device = pick_device(cfg.device)

    nca = build_model(cfg, args.img, device)
    nca.load_state_dict(torch.load(args.model, map_location=device))

    if args.seed is not None:
        torch.manual_seed(args.seed)
    steps = args.steps if args.steps is not None else cfg.inference.steps

    with torch.no_grad():
        state = nca.init_state(1)
        for _ in range(steps):
            state = nca.step(state)
        rgb = nca.to_rgb(state)[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        alive = (nca.alive_mask(state)[0, 0] > 0).cpu().numpy()
        state_np = state[0].cpu().numpy()

    logger.info('Ran %d steps on a %dx%d world; %d alive cells, %d state channels',
                steps, nca.height, nca.width, int(alive.sum()), state_np.shape[0])

    cell = tuple(args.cell) if args.cell else (nca.seed_y, nca.seed_x)
    if not (0 <= cell[0] < nca.height and 0 <= cell[1] < nca.width):
        parser.error('--cell must be inside the {}x{} world'.format(nca.height, nca.width))

    explorer = Explorer(state_np, rgb, alive, steps, channels=args.channels, metric=args.metric, cell=cell)

    if args.save:
        explorer.fig.savefig(args.save, dpi=150, bbox_inches='tight')
        logger.info('Saved dashboard to %s', args.save)
    else:
        explorer.plt.show()


if __name__ == '__main__':
    main()
