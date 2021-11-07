import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt


train_data = torchvision.datasets.MNIST('dataset', train=True, download=True)

train_data.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop(30)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=256)


class RAE(nn.Module):
    def __init__(self):
        super(RAE, self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 4), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_layer2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv_layer3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.conv_layer4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(2, 2), stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(1024)
        self.flatten_layer = nn.Flatten()
        self.encoding_layer = nn.Linear(1024, 16)

        self.decoding_layer1 = nn.Linear(16, 2 * 2 * 1024)
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(1024, 2, 2))
        self.bn5 = nn.BatchNorm2d(1024)
        self.convT_layer1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2))
        self.bn6 = nn.BatchNorm2d(512)
        self.convT_layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2))
        self.bn7 = nn.BatchNorm2d(256)
        self.convT_layer3 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(4, 4), stride=(2, 2))

    def encoder(self, x):
        layer1 = F.relu(self.bn1(self.conv_layer1(x)))
        layer2 = F.relu(self.bn2(self.conv_layer2(layer1)))
        layer3 = F.relu(self.bn3(self.conv_layer3(layer2)))
        layer4 = F.relu(self.bn4(self.conv_layer4(layer3)))
        return F.relu(self.encoding_layer(self.flatten_layer(layer4)))

    def decoder(self, z):
        layer1 = F.relu(self.bn5(self.unflatten_layer(self.decoding_layer1(z))))
        layer2 = F.relu(self.bn6(self.convT_layer1(layer1)))
        layer3 = F.relu(self.bn7(self.convT_layer2(layer2)))
        return self.convT_layer3(layer3)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def show_images(x, x_hat):
    for i in range(x.shape[0]):
        fig = plt.figure(figsize=(10, 6))

        fig.add_subplot(1, 2, 1)

        plt.imshow(x[i, 0, :, :])
        plt.axis('off')
        plt.title('x')

        fig.add_subplot(1, 2, 2)

        plt.imshow(x_hat.detach().numpy()[i, 0, :, :])
        plt.axis('off')
        plt.title('x_hat')

        fig.show()


def train_epoch(model, data_loader, optimizer):
    model.train()

    train_loss = 0

    for x, _ in data_loader:
        z, x_hat = model(x)
        loss = ((x - x_hat)**2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss / len(data_loader.dataset)


model = RAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):
    x, _ = next(iter(train_loader))
    z, x_hat = model(x)
    show_images(x, x_hat)
    print(z)
    train_loss = train_epoch(model, train_loader, optimizer)
    print('\n EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, epochs, train_loss))
