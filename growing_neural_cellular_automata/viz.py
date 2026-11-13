import os

import numpy as np
from PIL import Image


def _to_uint8(frame):
    return (np.clip(frame, 0.0, 1.0) * 255).round().astype(np.uint8)


def save_gif(frames, path, fps=20, scale=8):
    """Save frames (T, H, W, 3 floats in [0, 1]) as a looping GIF, upscaled with nearest-neighbour."""
    folder = os.path.dirname(os.fspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    images = []
    for frame in frames:
        img = Image.fromarray(_to_uint8(frame))
        if scale != 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=max(1, int(1000 / fps)), loop=0)


def play_frames(frames, fps=20):
    """Play precomputed frames in one matplotlib window, updating a single image in place."""
    import matplotlib.pyplot as plt  # imported here so training works without a display

    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(np.clip(frames[0], 0.0, 1.0), interpolation='nearest')
    for t, frame in enumerate(frames):
        im.set_data(np.clip(frame, 0.0, 1.0))
        ax.set_title('t = {}'.format(t))
        plt.pause(1.0 / fps)
    plt.show()
