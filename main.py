import random

import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

from models import rae_mnist
from data_loaders import data_loader


# CONFIG
DATASET_NAME = 'MNIST'
MODEL = rae_mnist.RAE_MNIST
DATA_LOADERS = data_loader.load_mnist_data
NUM_EPOCHS = 20
LOAD_MODEL_SNAPSHOT = True
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 32

torch.manual_seed(10)

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def show_images(x, x_hat):
    for i in range(x.shape[0]):
        fig = plt.figure(figsize=(10, 6))

        fig.add_subplot(1, 2, 1)

        plt.imshow(x.cpu()[i, 0, :, :])
        plt.axis('off')
        plt.title('x')

        fig.add_subplot(1, 2, 2)

        plt.imshow(x_hat.cpu().detach().numpy()[i, 0, :, :])
        plt.axis('off')
        plt.title('x_hat')

        fig.show()


def show_latent_features(model):
    model.eval()

    fig = plt.figure(figsize=(16, 16))
    for i in range(16):
        z = 10 * F.one_hot(torch.tensor(i), num_classes=16).float().reshape((1, 16)).to(device)
        x_hat = model.decoder(z)
        fig.add_subplot(4, 4, i+1)
        plt.imshow(x_hat.cpu().detach().numpy()[0, 0, :, :])
        plt.axis('off')
        plt.title('latent feature {}'.format(i+1))
    fig.show()


def show_morphing_effect(model, x1, x2, n=10):
    model.eval()

    fig = plt.figure(figsize=(2*(n+1), 2))
    z1, x1_hat = model(x1)
    z2, x2_hat = model(x2)

    fig.add_subplot(1, n+1, 1)
    plt.imshow(x1_hat.cpu().detach().numpy()[0, 0, :, :])
    plt.axis('off')

    for i in range(1, n):
        x_hat = model.decoder(torch.lerp(z1, z2, i/n))
        fig.add_subplot(1, n+1, i+1)
        plt.imshow(x_hat.cpu().detach().numpy()[0, 0, :, :])
        plt.axis('off')

    fig.add_subplot(1, n+1, n+1)
    plt.imshow(x2_hat.cpu().detach().numpy()[0, 0, :, :])
    plt.axis('off')

    fig.show()


def show_morphing_effect_of_samples(model, x, n=10):
    for i in range(x.shape[0] - 1):
        show_morphing_effect(model, x[i:i+1, :, :, :], x[i+1:i+2, :, :, :], n)



def train_epoch(model, device, data_loader, optimizer, beta, _lambda):
    model.train()

    train_loss = 0

    for x, _ in data_loader:
        x = x.to(device)
        z, x_hat = model(x)

        loss_rec = ((x - x_hat).square()).sum(axis=(2, 3)).mean()
        loss_rae = (z.square()).sum(axis=1).mean()
        loss_reg = sum(parameter.square().sum() for parameter in model.parameters())

        total_loss = loss_rec + beta * loss_rae + _lambda * loss_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    return train_loss / len(data_loader.dataset)


model = MODEL().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_epoch = 1

if LOAD_MODEL_SNAPSHOT:
    try:
        snapshot = torch.load('model_snapshots/{}_{}'.format(DATASET_NAME, MODEL.__name__))
        model.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        start_epoch = snapshot['epoch'] + 1
        print('Successfully loaded model from snapshot. Starting with epoch {}'.format(start_epoch))
    except:
        start_epoch = 1
        print('Could not load model snapshot. Starting with epoch 1')

train_loader, test_loader = DATA_LOADERS(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

test_x, _ = next(iter(test_loader))

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    test_x = test_x.to(device)
    test_z, test_x_hat = model(test_x)

    model.eval()

    show_latent_features(model)
    show_images(test_x, test_x_hat)
    x1_idx = random.randint(0, TEST_BATCH_SIZE)
    x2_idx = random.randint(0, TEST_BATCH_SIZE)
    show_morphing_effect_of_samples(model, test_x)

    model.train()

    train_loss = train_epoch(model, device, train_loader, optimizer, beta=1e-4, _lambda=1e-7)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'model_snapshots/{}_{}'.format(DATASET_NAME, MODEL.__name__))

    print('\n EPOCH {}/{} \t train loss {:.4f}'.format(epoch, start_epoch + NUM_EPOCHS - 1, train_loss))
