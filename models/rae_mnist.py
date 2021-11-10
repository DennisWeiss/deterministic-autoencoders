import torch.nn as nn
import torch.nn.functional as F


class RAE_MNIST(nn.Module):
    def __init__(self):
        super(RAE_MNIST, self).__init__()

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

        self.decoding_layer1 = nn.Linear(16, 6 * 6 * 1024)
        self.unflatten_layer = nn.Unflatten(dim=1, unflattened_size=(1024, 6, 6))
        self.bn5 = nn.BatchNorm2d(1024)
        self.convT_layer1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=(2, 2))
        self.bn6 = nn.BatchNorm2d(512)
        self.convT_layer2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(2, 2))
        self.bn7 = nn.BatchNorm2d(256)
        self.convT_layer3 = nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=(4, 4), stride=(1, 1))

    def encoder(self, x):
        layer1 = F.relu(self.bn1(self.conv_layer1(x)))
        layer2 = F.relu(self.bn2(self.conv_layer2(layer1)))
        layer3 = F.relu(self.bn3(self.conv_layer3(layer2)))
        layer4 = F.relu(self.bn4(self.conv_layer4(layer3)))
        encoding = F.relu(self.encoding_layer(self.flatten_layer(layer4)))
        return encoding

    def decoder(self, z):
        layer1 = F.relu(self.bn5(self.unflatten_layer(self.decoding_layer1(z))))
        layer2 = F.relu(self.bn6(self.convT_layer1(layer1)))
        layer3 = F.relu(self.bn7(self.convT_layer2(layer2)))
        output = self.convT_layer3(layer3)
        return output

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)