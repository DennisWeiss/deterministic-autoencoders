import torch.utils.data
import torchvision


def load_mnist_data(train_batch_size, test_batch_size):
    train_data = torchvision.datasets.MNIST('dataset/train', train=True, download=True)
    test_data = torchvision.datasets.MNIST('dataset/test', train=False, download=True)

    for data in [train_data, test_data]:
        data.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.CenterCrop(30)]
        )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, pin_memory=True)

    return train_loader, test_loader