from tqdm import tqdm

def Training(gen, disc, criterion, z_dim, n_epochs, dataloader, device):
    """
    :param n_epochs:
    :param dataloader:
    :param device:
    :return:
    """
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)
            # update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            #update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
