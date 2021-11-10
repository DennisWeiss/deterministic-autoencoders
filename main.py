from tqdm import tqdm

import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

from models import rae_mnist, rae_celeba
from data_loaders import data_loader


# CONFIG
DATASET_NAME = 'CelebA'
MODEL = rae_celeba.RAE_CelebA
DATA_LOADERS = data_loader.load_celeba_data
NUM_EPOCHS = 20
LOAD_MODEL_SNAPSHOT = True
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 1e-3
EMBEDDING_LOSS_WEIGHT = 1e-4
REGULARIZER_LOSS_WEIGHT = 1e-7


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

        plt.imshow(torch.transpose(torch.transpose(x[i, :, :, :], 1, 2), 0, 2).cpu())
        plt.axis('off')
        plt.title('x')

        fig.add_subplot(1, 2, 2)

        plt.imshow(torch.transpose(torch.transpose(torch.clip(x_hat[i, :, :, :], 1, 2), 0, 1), 0, 2).cpu().detach().numpy())
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


def show_morphing_effect_of_samples(model, x, n=10):
    for i in range(x.shape[0] - 1):
        show_morphing_effect(model, x[i:i+1, :, :, :], x[i+1:i+2, :, :, :], n)


def train_epoch(model, device, data_loader, optimizer, beta, _lambda):
    model.train()

    train_total_loss = 0
    train_loss_rec = 0
    train_loss_rae = 0
    train_loss_reg = 0

    for x, _ in tqdm(data_loader, desc='Epoch {}/{}'.format(epoch, start_epoch + NUM_EPOCHS - 1), unit='batch', colour='blue'):
        x = x.to(device)
        z, x_hat = model(x)

        loss_rec = ((x - x_hat).square()).sum(axis=(2, 3)).mean()
        loss_rae = (z.square()).sum(axis=1).mean()
        loss_reg = sum(parameter.square().sum() for parameter in model.parameters())

        total_loss = loss_rec + beta * loss_rae + _lambda * loss_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item()
        train_loss_rec += loss_rec.item()
        train_loss_rae += loss_rae.item()
        train_loss_reg += loss_reg.item()

    return train_total_loss / len(data_loader.dataset), train_loss_rec / len(data_loader.dataset), \
           train_loss_rae / len(data_loader.dataset), train_loss_reg / len(data_loader.dataset)


model = MODEL().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)

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

    # show_latent_features(model)
    # show_images(test_x, test_x_hat)
    show_morphing_effect_of_samples(model, test_x)

    model.train()

    train_total_loss, train_loss_rec, train_loss_rae, train_loss_reg = train_epoch(model, device, train_loader, optimizer, beta=EMBEDDING_LOSS_WEIGHT, _lambda=REGULARIZER_LOSS_WEIGHT)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'model_snapshots/{}_{}'.format(DATASET_NAME, MODEL.__name__))

    print('\n EPOCH {}/{} \t train: total loss {:.4f} \t loss_rec {:.4f} \t loss_rae {:.4f} \t loss_reg {:.4f}'.format(epoch, start_epoch + NUM_EPOCHS - 1, train_total_loss, train_loss_rec, train_loss_rae, train_loss_reg))
