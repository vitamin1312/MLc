import torch.optim as optim
import torch.nn as nn
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import imageio
from hyperparameters import *
from utils import get_transforms
import numpy as np
from matplotlib import pyplot as plt


class Learner():

    def __init__(self, generator, discriminator,
                 train_loader, transforms,
                 generator_lr=0.0002, discriminator_lr=0.0002):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.train_loader = train_loader
        transforms = get_transforms()
        # self.transform = transforms['transform']
        self.to_pil_image = transforms['to_pil_image']
        self.optim_g = optim.Adam(generator.parameters(), generator_lr)
        self.optim_d = optim.Adam(discriminator.parameters(), discriminator_lr)
        self.loss_fn = nn.BCELoss().to(device)
        self.losses_g = list()
        self.losses_d = list()
        self.images = list()

    
    @staticmethod
    def label_real(size):
        data = torch.ones(size, 1)
        return data.to(device)
    
    @staticmethod
    def label_fake(size):
        data = torch.zeros(size, 1)
        return data.to(device)
    
    @staticmethod
    def create_noise(sample_size, nz):
        return torch.randn(sample_size, nz).to(device)

    def train_discriminator(self, data_real, data_fake):
  
        b_size = data_real.size(0)
        real_label = self.label_real(b_size)
        fake_label = self.label_fake(b_size)

        self.optim_d.zero_grad()

        output_real = self.discriminator(data_real)
        loss_real = self.loss_fn(output_real, real_label)
        output_fake = self.discriminator(data_fake)
        loss_fake = self.loss_fn(output_fake, fake_label)

        loss_real.backward()
        loss_fake.backward()
        self.optim_d.step()

        return loss_real + loss_fake
    
    def train_generator(self, data_fake):

        b_size = data_fake.size(0)
        real_label = self.label_real(b_size)

        self.optim_g.zero_grad()

        output = self.discriminator(data_fake)
        loss = self.loss_fn(output, real_label)

        loss.backward()
        self.optim_g.step()
        
        return loss
    
    def train_models(self, epochs=epochs):

        noise = self.create_noise(sample_size, nz)
        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            loss_g = 0.0
            loss_d = 0.0
            for bi, data in tqdm(enumerate(self.train_loader)):
                image, _ = data
                image = image.to(device)
                b_size = len(image)
                # run the discriminator for k number of steps
                for step in range(k):
                    data_fake = self.generator(self.create_noise(b_size, nz)).detach()
                    data_real = image
                    # train the discriminator network
                    loss_d += self.train_discriminator(data_real, data_fake).item()
                data_fake = self.generator(self.create_noise(b_size, nz))
                # train the generator network
                loss_g += self.train_generator(data_fake).item()
            # create the final fake image for the epoch
            generated_img = self.generator(noise).cpu().detach()
            # make the images as grid
            generated_img = make_grid(generated_img)
            # save the generated torch tensor models to disk
            save_image(generated_img, f"../outputs/gen_img{epoch}.png")
            self.images.append(generated_img)
            epoch_loss_g = loss_g / bi # total generator loss for the epoch
            epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
            self.losses_g.append(epoch_loss_g)
            self.losses_d.append(epoch_loss_d)
            
            print(f"Epoch {epoch+1} of {epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
        
        print('DONE TRAINING')

    def save_results(self):
        torch.save(self.generator.state_dict(), '../outputs/generator.pth')
        imgs = [np.array(self.to_pil_image(img)) for img in self.images]
        imageio.mimsave('../outputs/generator_images.gif', imgs)

    def plot_learning(self):
        plt.figure()
        plt.plot(self.losses_g, label='Generator loss')
        plt.plot(self.losses_d, label='Discriminator Loss')
        plt.legend()
        plt.savefig('../outputs/loss.png')


