"""Growing MNIST digits with a neural cellular automaton.

Every training sample's target is a specific MNIST digit. Each batch is drawn from a pool of states. Each sample
starts either
  * from its pool state (the end of an earlier episode), or
  * from a freshly masked digit: part of the digit's ink is hidden and the cells must fill it in. The hidden
    fraction grows over training (curriculum).
Some samples then get a second stage in the same episode: one ink cell of the grown result is transplanted, at
the same position, onto an empty canvas and must regrow the same digit. Because both stages are in one graph,
the second stage's loss trains the first stage to write the digit's identity into the cells' hidden state, as
well as training the rule to read it back.

Digits are drawn as black ink on a transparent canvas: alpha = ink, RGB = 0 (premultiplied), so they appear
black on white.
"""
import logging
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from viz import save_gif

logger = logging.getLogger(__name__)

KINDS = ('pool', 'masked', 'cell')   # logged losses: stage 1 from a pool state / from a masked digit / stage 2
INK = 0.1   # alpha above which a pixel counts as ink (the same threshold as the live-cell mask)


def load_digits(data_dir, train=True, n_images=0, n_per_class=0, classes=None, device='cpu'):
    """MNIST digits as a (N, 28, 28) uint8 tensor on `device` (downloaded to data_dir on first use).

    classes restricts to those digit classes (None or empty = all 0-9). n_per_class then keeps the first
    n_per_class digits of each class, ordered by class; otherwise n_images keeps the first n_images digits
    (of the chosen classes) in dataset order. 0 means no limit.
    """
    from torchvision.datasets import MNIST
    dataset = MNIST(root=data_dir, train=train, download=True)
    digits, labels = dataset.data, dataset.targets
    classes = sorted(set(classes)) if classes else list(range(10))
    if n_per_class:
        keep = torch.cat([torch.nonzero(labels == c, as_tuple=True)[0][:n_per_class] for c in classes])
        digits, labels = digits[keep], labels[keep]
    else:
        keep = torch.isin(labels, torch.tensor(classes))
        digits, labels = digits[keep], labels[keep]
    if not n_per_class and n_images:
        digits, labels = digits[:n_images], labels[:n_images]
    counts = torch.bincount(labels, minlength=10).tolist()
    logger.info('Loaded %d %s digits; per class 0-9: %s', digits.shape[0], 'training' if train else 'test', counts)
    return digits.to(device)


def make_targets(nca, digits, idx):
    """(n, 4, H, W) premultiplied RGBA targets for digits[idx], centred in the world."""
    alpha = digits[idx].float() / 255.0
    n, h, w = alpha.shape
    targets = torch.zeros((n, 4, nca.height, nca.width), device=alpha.device)
    off_y, off_x = (nca.height - h) // 2, (nca.width - w) // 2
    targets[:, 3, off_y:off_y + h, off_x:off_x + w] = alpha
    return targets


def per_sample_loss(state, targets):
    return ((state[:, :4] - targets) ** 2).mean(dim=(1, 2, 3))


def mask_fraction(mcfg, i):
    """Fraction of ink hidden at iteration i (linear curriculum from mask_start to mask_end)."""
    if mcfg.curriculum_iters <= 0:
        return mcfg.mask_end
    t = min(1.0, i / mcfg.curriculum_iters)
    return mcfg.mask_start + (mcfg.mask_end - mcfg.mask_start) * t


@torch.no_grad()
def masked_start(nca, targets, frac):
    """Starting states showing each target digit with `frac` of its ink hidden.

    The hidden part is the ink nearest a random ink pixel, so it is one contiguous region. At least one ink
    pixel always stays visible. Visible cells get the target's RGBA; hidden channels start at zero.
    """
    n = targets.shape[0]
    state = nca.blank_state(n)
    for b in range(n):
        ys, xs = torch.nonzero(targets[b, 3] > INK, as_tuple=True)
        m = ys.numel()
        if m == 0:
            state[b] = nca.init_state(1)[0]   # empty digit (should not happen with MNIST): use the usual seed
            continue
        c = int(torch.randint(m, (1,)))
        dist = ((ys - ys[c]) ** 2 + (xs - xs[c]) ** 2).float()
        n_hide = min(m - 1, int(round(frac * m)))
        keep = torch.argsort(dist)[n_hide:]
        ky, kx = ys[keep], xs[keep]
        state[b, :4, ky, kx] = targets[b, :4, ky, kx]
    return state


@torch.no_grad()
def single_cell_start(nca, states):
    """Empty canvases each holding one cell copied (all channels, same position) from the matching state.

    The cell is a random ink cell (alpha > INK). Returns (new_states, ok); ok is False where a state had no such
    cell, and that canvas is left empty.
    """
    n = states.shape[0]
    out = nca.blank_state(n)
    ok = torch.zeros(n, dtype=torch.bool, device=states.device)
    for b in range(n):
        ys, xs = torch.nonzero(states[b, 3] > INK, as_tuple=True)
        if ys.numel() == 0:
            continue
        j = int(torch.randint(ys.numel(), (1,)))
        y, x = int(ys[j]), int(xs[j])
        out[b, :, y, x] = states[b, :, y, x]
        ok[b] = True
    return out, ok


def tile_rows(rows, gap=1, gap_value=0.6):
    """Tile a list of (T, B, H, W, 3) frame arrays into one (T, rows*H, B*W, 3) video, one row per array."""
    t, b, h, w, _ = rows[0].shape
    r = len(rows)
    out = np.full((t, r * h + (r - 1) * gap, b * w + (b - 1) * gap, 3), gap_value, dtype=np.float32)
    for i, arr in enumerate(rows):
        for j in range(b):
            out[:, i * (h + gap):i * (h + gap) + h, j * (w + gap):j * (w + gap) + w] = arr[:, j]
    return out


@torch.no_grad()
def demo_grid(nca, digits, frac, steps, n):
    """Grow the first n digits two ways and tile the result into one video.

    Rows: target | grown from the digit with `frac` of its ink masked | regrown from a single cell taken
    from the end of that run. Returns (frames (T, H, W, 3), {'masked': mse, 'cell': mse}).
    """
    n = digits.shape[0] if n <= 0 else min(n, digits.shape[0])
    idx = torch.arange(n, device=digits.device)
    targets = make_targets(nca, digits, idx)

    masked_frames, grown = nca.rollout_from(masked_start(nca, targets, frac), steps)
    cells, ok = single_cell_start(nca, grown)
    cell_frames, regrown = nca.rollout_from(cells, steps)

    target_rgb = nca.to_rgb(targets).permute(0, 2, 3, 1).cpu().numpy()
    target_frames = np.repeat(target_rgb[None], masked_frames.shape[0], axis=0)
    frames = tile_rows([target_frames, masked_frames, cell_frames])

    mse = {'masked': per_sample_loss(grown, targets).mean().item(),
           'cell': per_sample_loss(regrown, targets)[ok].mean().item() if bool(ok.any()) else float('nan')}
    return frames, mse


def pick_cell_mask(states):
    """(n, 1, H, W) float mask with a single 1 at a random ink cell (alpha > INK) of each state.

    Returns (mask, ok); ok is False where a state had no ink cell (its mask is all zero).
    """
    n, _, height, width = states.shape
    mask = torch.zeros((n, 1, height, width), device=states.device)
    ok = torch.zeros(n, dtype=torch.bool, device=states.device)
    for b in range(n):
        ys, xs = torch.nonzero(states[b, 3] > INK, as_tuple=True)
        if ys.numel() == 0:
            continue
        j = int(torch.randint(ys.numel(), (1,)))
        mask[b, 0, ys[j], xs[j]] = 1.0
        ok[b] = True
    return mask, ok


def train_mnist(nca, digits, test_digits, tcfg, mcfg, path):
    """Train on MNIST. Saves the weights to `path` at every evaluation and at the end, plus evaluation GIFs."""
    device = nca.conv_kernel.device
    path = Path(path)
    gif_dir = Path(tcfg.gif_dir) if tcfg.gif_dir else path.parent / (path.stem + '_gifs')
    batch, pool_size, n_digits = tcfg.batch_size, tcfg.pool_size, digits.shape[0]
    if pool_size < batch:
        raise ValueError('train.pool_size must be at least train.batch_size for MNIST training')
    if not 0.0 <= mcfg.mask_end < 1.0:
        raise ValueError('mnist.mask_end must be in [0, 1) so some ink stays visible')
    if mcfg.lr_decay_at and mcfg.lr_decay_at < mcfg.curriculum_iters:
        logger.warning('mnist.lr_decay_at (%d) is before the end of the curriculum (%d)',
                       mcfg.lr_decay_at, mcfg.curriculum_iters)

    for group in nca.optimizer.param_groups:
        group['lr'] = tcfg.lr
    scheduler = None
    if mcfg.lr_decay_at:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(nca.optimizer, milestones=[mcfg.lr_decay_at],
                                                         gamma=tcfg.lr_decay_factor)

    # Fill the pool with masked digits
    pool_idx = torch.randint(n_digits, (pool_size,), device=device)
    pool = torch.empty((pool_size, nca.state_dim, nca.height, nca.width), device=device)
    for s in range(0, pool_size, 256):
        pool[s:s + 256] = masked_start(nca, make_targets(nca, digits, pool_idx[s:s + 256]), mask_fraction(mcfg, 0))

    kind_sum, kind_count = [0.0] * len(KINDS), [0] * len(KINDS)
    losses = []
    start = time.perf_counter()

    logger.info('MNIST training for %d iterations on %d digits: world %dx%d, batch %d, pool %d, '
                'mask %.2f -> %.2f over %d iters, p(masked) %.2f, p(second stage) %.2f, lr %g (x%g at iter %d), '
                '%d hidden channels',
                tcfg.n_iter, n_digits, nca.height, nca.width, batch, pool_size, mcfg.mask_start, mcfg.mask_end,
                mcfg.curriculum_iters, mcfg.p_masked, mcfg.p_single_cell, tcfg.lr, tcfg.lr_decay_factor,
                mcfg.lr_decay_at, nca.state_dim - 4)

    for i in range(1, tcfg.n_iter + 1):
        frac = mask_fraction(mcfg, i)

        # Draw a batch from the pool and decide how each sample starts
        sel = torch.randperm(pool_size, device=device)[:batch]
        state = pool[sel]
        tidx = pool_idx[sel]
        masked = torch.rand(batch, device=device) < mcfg.p_masked
        second = torch.rand(batch, device=device) < mcfg.p_single_cell

        with torch.no_grad():
            rows = masked.nonzero(as_tuple=True)[0]
            if rows.numel():
                new = torch.randint(n_digits, (rows.numel(),), device=device)
                tidx[rows] = new
                state[rows] = masked_start(nca, make_targets(nca, digits, new), frac)

        targets = make_targets(nca, digits, tidx)

        nca.optimizer.zero_grad()

        # Stage 1: grow every sample
        steps = int(torch.randint(low=tcfg.n_steps[0], high=tcfg.n_steps[1], size=(1,)))
        for _ in range(steps):
            state = nca.step(state)
        loss1 = per_sample_loss(state, targets)
        final = state.detach().clone()

        # Stage 2: transplant one ink cell of the stage-1 result onto an empty canvas and regrow. The transplant
        # is a masked blend, so gradients reach the stage-1 rollout that produced the cell.
        loss2 = None
        rows2 = second.nonzero(as_tuple=True)[0]
        if rows2.numel():
            cell_mask, ok = pick_cell_mask(state.detach()[rows2])
            rows2, cell_mask = rows2[ok], cell_mask[ok]
        if rows2.numel():
            blank = nca.blank_state(rows2.numel())
            state2 = blank + (state[rows2] - blank) * cell_mask
            steps2 = int(torch.randint(low=tcfg.n_steps[0], high=tcfg.n_steps[1], size=(1,)))
            for _ in range(steps2):
                state2 = nca.step(state2)
            loss2 = per_sample_loss(state2, targets[rows2])
            final[rows2] = state2.detach()

        # Each stage-1 and stage-2 loss counts once, averaged over the batch
        loss = loss1.sum() / batch
        if loss2 is not None:
            loss = loss + loss2.sum() / batch
        loss.backward()

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            logger.error('Loss became non-finite at iter %d; stopping. Last saved weights: %s', i, path)
            break

        do_log = i == 1 or i % tcfg.log_every == 0 or i == tcfg.n_iter
        grad_norm = None
        if do_log:
            grad_norm = float(nn.utils.clip_grad_norm_(nca.perception.parameters(), float('inf')))

        if tcfg.normalize_grads:
            for p in nca.perception.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)

        nca.optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Write back the stage-2 result where there was one, otherwise the stage-1 result
        with torch.no_grad():
            pool[sel] = final
            pool_idx[sel] = tidx

        losses.append(loss_val)
        for m, v in zip(masked.tolist(), loss1.detach().tolist()):
            kind_sum[int(m)] += v
            kind_count[int(m)] += 1
        if loss2 is not None:
            for v in loss2.detach().tolist():
                kind_sum[2] += v
                kind_count[2] += 1

        if do_log:
            elapsed = time.perf_counter() - start
            rate = i / elapsed
            eta = (tcfg.n_iter - i) / rate
            window = losses[-tcfg.log_every:]
            per_kind = '  '.join('{} {:.5f}'.format(name, kind_sum[k] / kind_count[k]) if kind_count[k]
                                 else '{} -'.format(name) for k, name in enumerate(KINDS))
            logger.info('iter %d/%d | mask %.2f | loss %.5f (avg of last %d: %.5f) | per sample: %s | grad %.3g | '
                        'lr %.2g | %.2f it/s | elapsed %s | eta %s',
                        i, tcfg.n_iter, frac, loss_val, len(window), sum(window) / len(window), per_kind,
                        grad_norm, nca.optimizer.param_groups[0]['lr'], rate,
                        timedelta(seconds=int(elapsed)), timedelta(seconds=int(eta)))
            kind_sum, kind_count = [0.0] * len(KINDS), [0] * len(KINDS)

        if tcfg.eval_every and (i % tcfg.eval_every == 0 or i == tcfg.n_iter):
            frames, mse = demo_grid(nca, test_digits, frac, tcfg.eval_steps, mcfg.eval_digits)
            gif_path = gif_dir / 'iter_{:06d}.gif'.format(i)
            save_gif(frames, gif_path, fps=tcfg.gif_fps, scale=mcfg.gif_scale)
            torch.save(nca.state_dict(), path)
            logger.info('eval @ iter %d on %d test digits (mask %.2f, %d steps): MSE from masked %.5f, '
                        'from single cell %.5f -> %s', i,
                        test_digits.shape[0] if mcfg.eval_digits <= 0 else min(mcfg.eval_digits, test_digits.shape[0]), frac,
                        tcfg.eval_steps, mse['masked'], mse['cell'], gif_path)

    torch.save(nca.state_dict(), path)
    logger.info('Done in %s; weights saved to %s', timedelta(seconds=int(time.perf_counter() - start)), path)
    return losses
