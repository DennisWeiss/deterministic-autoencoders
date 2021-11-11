from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transf

from models import rae_celeba
from util import util

IMG1 = 'toto_wolff.jpg'
IMG2 = 'christian_horner.jpg'

MODEL = rae_celeba.RAE_CelebA
DATASET = 'CelebA'
USE_CUDA_IF_AVAILABLE = True

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

x1 = transf.Resize(65)(transf.CenterCrop(218)(transf.ToTensor()(Image.open('assets/{}'.format(IMG1))))).unsqueeze(0).to(device)
x2 = transf.Resize(65)(transf.CenterCrop(218)(transf.ToTensor()(Image.open('assets/{}'.format(IMG2))))).unsqueeze(0).to(device)

model = MODEL().to(device)

model.eval()

snapshot = torch.load('model_snapshots/{}_{}'.format(DATASET, MODEL.__name__))
model.load_state_dict(snapshot['model_state_dict'])

print(x1.shape)

z1 = model.encoder(x1)
z2 = model.encoder(x2)

x_hat1 = model.decoder(z1)
x_hat2 = model.decoder(z2)


def show_img(img):
    plt.imshow(torch.transpose(torch.transpose(torch.clip(img, 0, 1), 2, 3), 1, 3)[0, :, :, :].cpu().detach().numpy())
    plt.axis('off')
    plt.show()


show_img(x1)
show_img(x2)
show_img(x_hat1)
show_img(x_hat2)

util.show_morphing_effect(model, x1, x2)

