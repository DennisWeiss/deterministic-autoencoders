import torch
import matplotlib.pyplot as plt


def show_morphing_effect(model, x1, x2, n=10):
    model.eval()

    fig = plt.figure(figsize=(2*(n+1), 2))
    z1, x1_hat = model(x1)
    z2, x2_hat = model(x2)

    fig.add_subplot(1, n+1, 1)
    plt.imshow(torch.transpose(torch.transpose(torch.clip(x1_hat, 0, 1), 2, 3), 1, 3).cpu().detach().numpy()[0, :, :, :])
    plt.axis('off')

    for i in range(1, n):
        x_hat = model.decoder(torch.lerp(z1, z2, i/n))
        fig.add_subplot(1, n+1, i+1)
        plt.imshow(torch.transpose(torch.transpose(torch.clip(x_hat, 0, 1), 2, 3), 1, 3).cpu().detach().numpy()[0, :, :, :])
        plt.axis('off')

    fig.add_subplot(1, n+1, n+1)
    plt.imshow(torch.transpose(torch.transpose(torch.clip(x2_hat, 0, 1), 2, 3), 1, 3).cpu().detach().numpy()[0, :, :, :])
    plt.axis('off')

    fig.show()