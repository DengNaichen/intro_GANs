from torch import nn


class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector(int), for generator, noise vector is the input
        im_dim: the dimension of the images (MNIST 784), which is the output of generator
        hidden_dim: the inner dimension, a scalar
        here the generator is the simplest FNN
    """

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            # input layer, the input is noise(not conditional GANs, so only noise)
            self.get_generator_block(z_dim, hidden_dim),
            # hidden layers
            self.get_generator_block(hidden_dim, hidden_dim * 2),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # output layer, the output is an image
            nn.Linear(hidden_dim * 8, im_dim),
            # todo: why need a sigmoid here
            nn.Sigmoid()
        )

    def get_generator_block(self, input_dim, output_dim):
        """
        :param input_dim: scalar
        :param output_dim: scalar
        :return: a single layer
        """
        return nn.Sequential(
            # linear function
            nn.Linear(input_dim, output_dim),
            # todo: not sure why need a batch normalization here, even not sure what bn do.
            # see detail here:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d
            nn.BatchNorm1d(output_dim),
            # activation function
            nn.ReLU(inplace=True)
        )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return self.gen(noise)

    def get_gen(self):
        """
        Returns:
            the sequential model
        """
        return self.gen
