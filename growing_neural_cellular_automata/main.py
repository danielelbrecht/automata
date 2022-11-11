import torch
from model import NeuralCellularAutomata
import cv2 as cv
from torchvision.io import read_image, ImageReadMode

import matplotlib.pyplot as plt

def main():

    # TODO - separate interal state from external state
    # TODO - Look ar representations in state and mlp layers

    img = read_image('images/lizard-emoji_40.png', ImageReadMode.RGB).to(torch.float32)
    img = img / 255.

    nca = NeuralCellularAutomata(img, internal_dim=14, mlp_dim=128)
    nca.load_state_dict(torch.load('best_model3.pt'))
    #nca.train_ca(n_iter=5000, batch_size=1)
    #exit()

    # Infer with trained model
    steps = 100
    state = nca.init_state()

    for i in range(steps):
        img = nca.get_img_state(state)
        live = nca.get_live_cell_state(state)

        plt.imshow(img)
        plt.title('t = {}'.format(i))
        plt.pause(0.1)
        plt.draw()


        # Update state
        state = nca.step(state)


if __name__ == '__main__':
    main()