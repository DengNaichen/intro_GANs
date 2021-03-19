from torch import nn


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
        # inplace = True: inplace=True means that it will modify the input directly, without allocating any
        # additional output. It can sometimes slightly decrease the memory usage, but may not always be a valid
        # operation (because the original input is destroyed). However, if you don’t see an error, it means that your
        # use case is valid.
    )


class Discriminator(nn.Module):

    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
            # todo: no activation function? and why need one more linear layer?
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc
