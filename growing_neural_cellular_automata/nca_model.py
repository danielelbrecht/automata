import logging
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnnf

from viz import save_gif

logger = logging.getLogger(__name__)


class NeuralCellularAutomata(nn.Module):

    def __init__(self, target_image, internal_dim=14, mlp_dim=128, pmask=0.5, lr=2e-3,
                 norm='none', rgba=True, world_scale=2):
        """
        target_image: (4, H, W) RGBA or (3, H, W) RGB in [0, 1].
        norm: 'none' (original paper) or 'layernorm' (global LayerNorm over (C, H, W), the old behaviour).
        world_scale: the grid is world_scale times larger than the image in each direction. The target is
                     pasted unchanged into the centre of an otherwise empty canvas, so the loss also
                     penalises growth outside the image and the cells never see the edge of the grid.
        rgba: True  -> transparent (all zero) canvas, seed cell has alpha and hidden channels = 1,
                       loss on premultiplied RGBA, output composited over white.
              False -> old behaviour: white RGB canvas, black seed with alpha 1, loss on RGB only.
        """

        super().__init__()

        if norm not in ('none', 'layernorm'):
            raise ValueError("norm must be 'none' or 'layernorm', got {!r}".format(norm))
        self.norm = norm
        self.rgba = rgba

        # Build the training target
        if target_image.shape[0] == 4:
            rgb, alpha = target_image[:3], target_image[3:4]
        else:
            rgb, alpha = target_image, torch.ones_like(target_image[:1])
        if rgba:
            target = torch.cat([rgb * alpha, alpha], dim=0)  # premultiplied RGBA
        else:
            target = rgb
        target = target.contiguous()

        # Paste the target into the centre of a larger empty world
        channels, img_h, img_w = target.shape
        world_h, world_w = img_h * world_scale, img_w * world_scale
        canvas = torch.zeros((channels, world_h, world_w)) if rgba else torch.ones((channels, world_h, world_w))
        off_y, off_x = (world_h - img_h) // 2, (world_w - img_w) // 2
        canvas[:, off_y:off_y + img_h, off_x:off_x + img_w] = target
        target = canvas
        self.seed_y = off_y + img_h // 2   # the seed sits at the centre of the image
        self.seed_x = off_x + img_w // 2
        self.damage_scale = min(img_h / world_h, img_w / world_w)  # keeps damage sized relative to the image

        # Non-persistent buffers follow .to(device) but stay out of the state dict
        self.register_buffer('target_image', target, persistent=False)
        self.state_dim = 4 + internal_dim
        self.height, self.width = target.shape[1], target.shape[2]

        # Initialize Convolutional kernel
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32, requires_grad=False)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32, requires_grad=False)

        identity = torch.tensor([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=torch.float32, requires_grad=False)

        kernel = torch.stack([sobel_x / 8.0, sobel_y / 8.0, identity], dim=0)  # Sobel scaled by 1/8 as in the paper
        kernel = torch.unsqueeze(kernel, dim=1)
        kernel = [kernel for i in range(self.state_dim)]
        kernel = torch.cat(kernel, dim=0)
        self.register_buffer('conv_kernel', kernel, persistent=False)
        self.register_buffer('_blank', self.blank_state(1), persistent=False)  # what dead cells are reset to

        # Per-cell update network. The Identity keeps the layer indices (and so checkpoint keys)
        # the same whether or not the LayerNorm is used. LayerNorm has its own bias, so the first
        # conv only gets one when there is no norm.
        conv_in = nn.Conv2d(in_channels=self.state_dim * 3, out_channels=mlp_dim, stride=1,
                            kernel_size=1, bias=(norm == 'none'))
        if norm == 'layernorm':
            norm_layer = nn.LayerNorm((mlp_dim, self.height, self.width))
        else:
            norm_layer = nn.Identity()
        conv_out = nn.Conv2d(in_channels=mlp_dim, out_channels=self.state_dim, stride=1, kernel_size=1, bias=False)
        conv_out.weight.data.fill_(0)  # start as "do nothing"
        self.perception = nn.Sequential(conv_in, norm_layer, nn.ReLU(), conv_out)

        # Model training parameters
        self.optimizer = torch.optim.Adam(self.perception.parameters(), lr=lr)

        # Cellular automata params
        self.pmask = pmask

    # ------------------------------------------------------------------ states

    def blank_state(self, batch_size=1):
        """An empty canvas: all zeros (transparent) in rgba mode, white RGB in the old mode."""
        state = torch.zeros(size=(batch_size, self.state_dim, self.height, self.width),
                            device=self.conv_kernel.device)  # R, G, B, alpha, internal_dim
        if not self.rgba:
            state[:, :3] = 1.0
        return state

    def init_state(self, batch_size=1):
        state = self.blank_state(batch_size)

        if self.rgba:
            # Paper-style seed: every non-RGB channel (alpha + hidden) is 1 at the centre cell
            state[:, 3:, self.seed_y, self.seed_x] = 1.0
        else:
            state[:, :3, self.seed_y, self.seed_x] = 0   # black centre pixel
            state[:, 3, self.seed_y, self.seed_x] = 1.0  # alive

        return state

    def to_rgb(self, state):
        """(B, C, H, W) state -> (B, 3, H, W) image. In rgba mode the premultiplied RGBA is composited over white."""
        rgb = state[:, :3]
        if self.rgba:
            rgb = 1.0 - state[:, 3:4] + rgb
        return rgb

    def get_img_state(self, state):
        return self.to_rgb(state)[0].permute(1, 2, 0).detach().cpu().numpy()

    @staticmethod
    def get_live_cell_state(state):
        return state[0, 3, :, :].detach().cpu().numpy()

    def damage_masks(self, n):
        """(n, 1, H, W) bool masks, True inside a random circle. Centre in [-0.5, 0.5] and radius in [0.1, 0.4],
        in units of the image's half-size, so the damage is the same size relative to the image whatever the world size."""
        device = self.conv_kernel.device
        ys = torch.linspace(-1.0, 1.0, self.height, device=device).view(1, -1, 1)
        xs = torch.linspace(-1.0, 1.0, self.width, device=device).view(1, 1, -1)
        center = (torch.rand(2, n, 1, 1, device=device) - 0.5) * self.damage_scale
        radius = (torch.rand(n, 1, 1, device=device) * 0.3 + 0.1) * self.damage_scale
        x = (xs - center[0]) / radius
        y = (ys - center[1]) / radius
        return ((x * x + y * y) < 1.0).unsqueeze(1)

    # ------------------------------------------------------------------ dynamics

    @staticmethod
    def alive_mask(state):
        """(B, 1, H, W) float mask: 1 where a cell or any of its 3x3 neighbours has alpha > 0.1."""
        return (tnnf.max_pool2d(state[:, 3:4], kernel_size=3, stride=1, padding=1) > 0.1).to(state.dtype)

    def step(self, state_in):

        pre_live = self.alive_mask(state_in)

        # Run perception model
        x = tnnf.conv2d(state_in, self.conv_kernel, groups=self.state_dim, padding=1)
        x = self.perception(x)

        # Stochastic masking
        mask = torch.rand(size=(state_in.shape[0], 1, self.height, self.width), device=state_in.device)
        mask = (mask > self.pmask).to(torch.float32)

        x *= mask

        # Apply update
        state_out = state_in + x

        # Live cell masking: a cell must be alive both before and after the update, otherwise it is
        # reset to the blank canvas (all channels, not just the update).
        life = pre_live * self.alive_mask(state_out)
        state_out = state_out * life + self._blank * (1.0 - life)

        return state_out

    @torch.no_grad()
    def rollout(self, steps, batch_size=1):
        """Run the CA from the seed for `steps` steps; returns RGB frames as a (steps + 1, H, W, 3) array."""
        state = self.init_state(batch_size)
        frames = [self.to_rgb(state)[0]]
        for _ in range(steps):
            state = self.step(state)
            frames.append(self.to_rgb(state)[0])
        # Single device -> host transfer at the end instead of one per step
        return torch.stack(frames).permute(0, 2, 3, 1).cpu().numpy()

    @torch.no_grad()
    def rollout_from(self, state, steps):
        """Run the CA from a given (B, C, H, W) state.

        Returns (frames, final_state): frames is a (steps + 1, B, H, W, 3) numpy array of RGB images.
        """
        frames = [self.to_rgb(state)]
        for _ in range(steps):
            state = self.step(state)
            frames.append(self.to_rgb(state))
        return torch.stack(frames).permute(0, 1, 3, 4, 2).cpu().numpy(), state

    # ------------------------------------------------------------------ training

    def per_sample_loss(self, state):
        """MSE against the target (premultiplied RGBA, or RGB in the old mode), one value per batch element."""
        diff = state[:, :self.target_image.shape[0]] - self.target_image
        return (diff ** 2).mean(dim=(1, 2, 3))

    def train_ca(self, cfg, path='model.pt'):
        """Train with a TrainConfig.

        Saves the best-loss checkpoint to `path`, the latest weights to `<name>_last<ext>` at every
        inference run and at the end, and periodic inference GIFs.
        """
        path = Path(path)
        last_path = path.with_name(path.stem + '_last' + path.suffix)
        gif_dir = Path(cfg.gif_dir) if cfg.gif_dir else path.parent / (path.stem + '_gifs')

        # What the target looks like on screen, for comparing against rollout frames
        t = self.target_image
        target_vis = ((t[:3] + 1.0 - t[3:4]) if self.rgba else t).permute(1, 2, 0).cpu().numpy()

        for group in self.optimizer.param_groups:
            group['lr'] = cfg.lr
        scheduler = None
        if cfg.lr_decay_at:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[cfg.lr_decay_at],
                                                             gamma=cfg.lr_decay_factor)

        use_pool = cfg.pool_size > 0
        if use_pool:
            pool = self.init_state(1).repeat(cfg.pool_size, 1, 1, 1)
        n_damage = min(cfg.damage_n, cfg.batch_size - 1) if use_pool else 0

        best_loss = float('inf')
        best_iter = 0
        best_img = None
        losses = []
        start = time.perf_counter()

        logger.info('World %dx%d (image %dx%d at the centre)', self.height, self.width,
                    round(self.height * self.damage_scale), round(self.width * self.damage_scale))
        logger.info('Training for %d iterations: batch %d, %d-%d steps per rollout, lr %g (x%g at iter %d), '
                    'grad normalisation %s, pool %d, damage %d, norm=%s, rgba=%s',
                    cfg.n_iter, cfg.batch_size, cfg.n_steps[0], cfg.n_steps[1] - 1, cfg.lr, cfg.lr_decay_factor,
                    cfg.lr_decay_at, 'on' if cfg.normalize_grads else 'off',
                    cfg.pool_size, n_damage, self.norm, self.rgba)

        for i in range(1, cfg.n_iter + 1):

            if use_pool:
                # Draw a batch from the pool, put the worst sample back to the seed, damage the best few
                idx = torch.randperm(cfg.pool_size, device=pool.device)[:cfg.batch_size]
                state = pool[idx]
                with torch.no_grad():
                    order = self.per_sample_loss(state).argsort(descending=True)
                idx = idx[order]
                state = state[order]
                state[0] = self.init_state(1)[0]
                if n_damage > 0:
                    state[-n_damage:] = torch.where(self.damage_masks(n_damage),
                                                    self.blank_state(n_damage), state[-n_damage:])
            else:
                state = self.init_state(cfg.batch_size)

            self.optimizer.zero_grad()

            steps = int(torch.randint(low=cfg.n_steps[0], high=cfg.n_steps[1], size=(1,)))
            for _ in range(steps):
                state = self.step(state)

            # Compute loss
            loss = self.per_sample_loss(state).mean()
            loss.backward()

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                logger.error('Loss became non-finite at iter %d; stopping. Last saved weights: %s', i, last_path)
                break

            do_log = i == 1 or i % cfg.log_every == 0 or i == cfg.n_iter
            grad_norm = None
            if do_log:
                # max_norm=inf measures the gradient norm (before any normalisation) without clipping
                grad_norm = float(nn.utils.clip_grad_norm_(self.perception.parameters(), float('inf')))

            if cfg.normalize_grads:
                for p in self.perception.parameters():
                    if p.grad is not None:
                        p.grad /= (p.grad.norm() + 1e-8)

            # Apply update
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if use_pool:
                pool[idx] = state.detach()

            losses.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_iter = i
                best_img = self.get_img_state(state)
                torch.save(self.state_dict(), path)

            if do_log:
                elapsed = time.perf_counter() - start
                rate = i / elapsed
                eta = (cfg.n_iter - i) / rate
                window = losses[-cfg.log_every:]
                live = state[:, 3].sum().item() / state.shape[0]
                logger.info('iter %d/%d | loss %.5f (avg of last %d: %.5f) | best %.5f @ %d | grad %.3g | lr %.2g | '
                            'live %.0f | %.2f it/s | elapsed %s | eta %s',
                            i, cfg.n_iter, loss_val, len(window), sum(window) / len(window), best_loss, best_iter,
                            grad_norm, self.optimizer.param_groups[0]['lr'], live, rate,
                            timedelta(seconds=int(elapsed)), timedelta(seconds=int(eta)))

            if cfg.eval_every and (i % cfg.eval_every == 0 or i == cfg.n_iter):
                frames = self.rollout(cfg.eval_steps)
                gif_path = gif_dir / 'iter_{:06d}.gif'.format(i)
                save_gif(frames, gif_path, fps=cfg.gif_fps, scale=cfg.gif_scale)
                torch.save(self.state_dict(), last_path)
                final_mse = float(np.mean((np.clip(frames[-1], 0.0, 1.0) - target_vis) ** 2))
                logger.info('inference @ iter %d: %d steps, final-frame MSE %.5f -> %s',
                            i, cfg.eval_steps, final_mse, gif_path)

        torch.save(self.state_dict(), last_path)
        logger.info('Done in %s. Best loss %.5f at iter %d (saved to %s); latest weights saved to %s',
                    timedelta(seconds=int(time.perf_counter() - start)), best_loss, best_iter, path, last_path)

        return best_img, losses
