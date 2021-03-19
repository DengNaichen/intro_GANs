import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device='cpu'):
    """
    :param n_samples:
    :param noise_dim:
    :param device:
    :return:
    """
    return torch.randn(n_samples, noise_dim, device=device)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    :param gen:
    :param disc:
    :param criterion:
    :param real:
    :param num_images:
    :param z_dim:
    :param device:
    :return:
    """
    noise = get_noise(num_images, z_dim, device=device)
    fake = gen(noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    # You need a 'ground truth' tensor in order to calculate the loss.
    # the ground truth of fake is zero
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    # the ground truth of real is one
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    # the loss is average loss of fake and real

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    :param gen:
    :param disc:
    :param criterion:
    :param num_images:
    :param z_dim:
    :param device:
    :return:
    todo：why we need get generator loss?
    """
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss
