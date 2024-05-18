import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.lin1 = nn.Linear(self.nz, 7*7*256)
        self.bn1 = nn.BatchNorm2d(7*7*256)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv_transp2 = nn.ConvTranspose2d(256, 128, 5)
        self.bn2 = nn.BatchNorm2d(7*7*256)

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3), # 26
            nn.Conv2d(32, 64, 3), # 24
            nn.MaxPool2d(2,2), # 12
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, 3), # 10
            nn.MaxPool2d(2,2), # 5
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32*5*5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)