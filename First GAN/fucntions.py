import torch


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
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images.
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a
    #            'ground truth' tensor in order to calculate the loss.
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
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
