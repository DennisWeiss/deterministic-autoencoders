import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture

from data_loaders import data_loader
from models import rae_celeba

MODEL = rae_celeba.RAE_CelebA
DATA_LOADERS = data_loader.load_celeba_data
DATASET = 'CelebA'
USE_CUDA_IF_AVAILABLE = False

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def show_image(x):
    fig = plt.figure(figsize=(20, 20))
    for i in range(x.shape[0]):

        fig.add_subplot(5, 5, i+1)

        plt.imshow(torch.transpose(torch.transpose(torch.clip(x[i, :, :, :], 0, 1), 1, 2), 0, 2).cpu().detach().numpy())
        plt.axis('off')

    fig.show()


model = MODEL().to(device)

model.eval()

snapshot = torch.load('model_snapshots/{}_{}'.format(DATASET, MODEL.__name__))
model.load_state_dict(snapshot['model_state_dict'])

gmm_train_loader, gmm_test_loader = DATA_LOADERS(5_000, 64)
gmm_train_data, _ = next(iter(gmm_train_loader))
gmm_train_data = gmm_train_data.to(device)
gmm_z, _ = model(gmm_train_data)
gmm_z = gmm_z.detach().numpy()
print(gmm_z.shape)
gm = GaussianMixture(n_components=10).fit(gmm_z)

latent_samples = torch.from_numpy(gm.sample(25)[0]).float()

gmm_test_data, _ = next(iter(gmm_test_loader))
gmm_test_data = gmm_test_data.to(device)
gmm_test_z, gmm_test_x_hat = model(gmm_test_data)
generated_samples = model.decoder(latent_samples)



show_image(generated_samples)