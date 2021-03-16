from torch import nn


def get_generator_block(input_dim, output_dim):
    """
    :param input_dim: scalar,
    :param output_dim:
    :return: for single layer??
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class generator(nn.Module):

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        """
        :param z_dim: dimension of noise vector
        :param im_dim: dimension of image, which is the output
            dimension of generator
        :param hidden_dim:
        """
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen