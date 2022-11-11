import torch
import torch.nn as nn
import torch.nn.functional as tnnf


class NeuralCellularAutomata(nn.Module):

    def __init__(self, target_image, internal_dim, mlp_dim):

        super().__init__()

        self.target_image = target_image
        self.state_dim = 4 + internal_dim
        self.state = self.init_state()

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

        kernel = torch.stack([sobel_x, sobel_y, identity], dim=0)
        kernel = torch.unsqueeze(kernel, dim=1)
        kernel = [kernel for i in range(self.state_dim)]
        kernel = torch.cat(kernel, dim=0)
        self.conv_kernel = kernel
        self.conv_kernel.requires_grad = False

        # Initialize
        conv = nn.Conv2d(in_channels=mlp_dim, out_channels=self.state_dim, stride=1, kernel_size=1, bias=False)
        conv.weight.data.fill_(0)
        self.perception = nn.Sequential(nn.Conv2d(in_channels=self.state_dim * 3, out_channels=mlp_dim, stride=1, kernel_size=1, bias=False),
                                        nn.LayerNorm((mlp_dim, 40, 40)),
                                        nn.ReLU(),
                                        conv)

        # Model training parameters
        self.optimizer = torch.optim.Adam(self.perception.parameters(), lr=0.00005)
        self.loss_func = torch.nn.MSELoss()

        # Cellular automata params
        self.pmask = 0.5

    def init_state(self, batch_size=1):

        state = torch.zeros(size=(batch_size, self.state_dim, self.target_image.shape[1], self.target_image.shape[2]))  # R, G, B, alpha, internal_dim
        #state[:, :3, :, :, ] += torch.ones(size=(1, 3, self.target_image.shape[1], self.target_image.shape[2]))  # Set background of image channels to white
        state[:, :3, :, :, ] += 1.0

        # Set center to black
        center_x = self.target_image.shape[1] // 2
        center_y = self.target_image.shape[2] // 2
        state[:, :3, center_x, center_y] = 0

        # Set center alpha position to 1.0
        state[:, 3, center_x, center_y] = 1.0

        return state

    @staticmethod
    def get_img_state(state):
        return state[0, :3, :, :].permute(1, 2, 0).detach().cpu().numpy()

    @staticmethod
    def get_live_cell_state(state):
        return state[0, 3, :, :].detach().cpu().numpy()

    def step(self, state_in):

        # Run perception model
        x = tnnf.conv2d(state_in, self.conv_kernel, groups=self.state_dim, padding=1)

        #rint(x.shape)

        #print(torch.unique(x[0, 2]))
        #print(torch.sum(x[0, 2]))
        #exit()


        x = self.perception(x)

        # Stochastic masking
        mask = torch.rand(size=(self.target_image.shape[1], self.target_image.shape[2]))
        mask = (mask > self.pmask).to(torch.float32)

        x *= mask

        # Live cell masking
        live = tnnf.max_pool2d(state_in[:, 3, :, :], kernel_size=(3, 3), stride=1, padding=1)  # Just looking at alpha layer
        live = (live > 0.1).to(torch.float32)
        #live = torch.squeeze(live)
        live = torch.unsqueeze(live, dim=1)

        #print(x.shape)
        #print(live.shape)
        #exit()

        x *= live

        # Apply update
        state_out = state_in + x

        # Clip first 4 channels to [0, 1] range
        #state_out[:, :4, :, :] = torch.clip(state_out[:, :4, :, :], min=0, max=1)

        return state_out

    def train_ca(self, n_iter, n_steps=(64, 96), batch_size=8):

        best_loss = torch.inf
        losses = []

        for i in range(n_iter):

            # Re-initialize state
            state = self.init_state(batch_size)

            self.optimizer.zero_grad()

            steps = torch.randint(low=n_steps[0], high=n_steps[1], size=(1,))

            #intermediate_steps = []
            for step in range(steps):
                state = self.step(state)
                #intermediate_steps.append(state)


            # Compute loss
            loss = self.loss_func(state[0, :3, :, :], self.target_image)  # Compare produced image to target image
            loss.backward()

            losses.append(loss.detach().cpu().numpy())

            # Apply update
            self.optimizer.step()

            print('Iter: {}  Loss: {}  Live: {}'.format(i, loss, torch.sum(state[0, 3, :, :])))

            if loss < best_loss:
                best_loss = loss
                best_img = self.get_img_state(state)

                torch.save(self.state_dict(), 'best_model3.pt')


        return best_img, losses

