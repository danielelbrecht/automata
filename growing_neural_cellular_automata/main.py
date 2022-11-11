import torch
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
import argparse

# Local imports
from nca_model import NeuralCellularAutomata

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='XVision detection')
    parser.add_argument('-t', '--train', help='Do training', action='store_true')
    parser.add_argument('-e', '--eval', help='Do evaluation', action='store_true')
    parser.add_argument('--model', type=str, help='Path to save or load model from')
    parser.add_argument('--img', type=str, help='Path to desired image for training')


    args = parser.parse_args()
    train = args.train
    eval = args.eval
    model_path = args.model
    img_path = args.img

    img = read_image(img_path, ImageReadMode.RGB).to(torch.float32)
    img = img / 255.

    nca = NeuralCellularAutomata(img, internal_dim=14, mlp_dim=128)


    if train:

        nca.train_ca(n_iter=5000, batch_size=1, path=model_path)
        #exit()


    if eval:

        # Infer with trained model
        steps = 100

        # Initialize model
        nca.load_state_dict(torch.load(model_path))
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