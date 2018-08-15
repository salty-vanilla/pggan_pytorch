import torch
import time
import os
import numpy as np
from PIL import Image
from generator import Generator
from disciriminator import Discriminator
from data_loader import Dataset


class PGGAN:
    def __init__(self, input_dim=100, 
                 nb_growing=8,
                 upsampling='subpixel',
                 downsampling='avg_pool',
                 device='cuda',
                 lambda_=10.):
        self.input_dim = input_dim
        self.nb_growing = nb_growing
        self.device = device
        self.lambda_ = lambda_
        self.generator = Generator(input_dim=input_dim,
                                   nb_growing=nb_growing,
                                   upsampling=upsampling)
        self.discriminator = Discriminator(nb_growing=nb_growing,
                                           downsampling=downsampling)
        self.resolutions = [(2**(2+i), 2**(2+i)) for i in range(nb_growing)]   
        self.z_sampler = torch.distributions.Normal(0., 1.)

    def fit(self, image_dir,
            nb_epoch=100,
            batch_size=32,
            lr_d=2e-4,
            lr_g=2e-4,
            logdir='logs',
            save_steps=10,
            visualize_steps=1):

        self.discriminator.train()
        self.generator.train()
        fixed_z = self.z_sampler.sample((batch_size, self.input_dim))
        for growing_step in range(self.nb_growing):
            current_logdir = os.path.join(logdir, 'growing_step_%d' % growing_step+1)
            os.makedirs(current_dirs, exist_ok=True)

            dataset = Dataset(target_size=self.resolutions[growing_step])
            data_loader = dataset.flow_from_directory(image_dir, 
                                                      batch_size=batch_size)
            
            opt_d = optim.Adam(self.discriminator.parameters(), lr_d,
                               betas=(0.5, 0.999)) 
            opt_g = optim.Adam(self.generator.parameters(), lr_g,
                               betas=(0.5, 0.999))
            print('\n', '='*10, 
                  '\nGrowing Step %d / %d\n' % (growing_step+1, self.nb_growing),
                  '='*10)
            for epoch in range(1, nb_epoch+1):
                print('Epoch %d / %d' % (epoch, nb_epoch))
                start = time.time()
                for iter_, x in data_loader:
                    bs = x_real.shape[0]

                    # update discriminator
                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                    for p in self.generator.parameters():
                        p.requires_grad = False

                    self.discriminator.zero_grad()
                    x_real = x.to(self.device)
                    d_x_real = self.discriminator(x_real, 
                                                  growing_step=growing_step)
                    d_x_real = -d_x_real.mean()
                    d_x_real.backward()

                    z = self.z_sampler.sample((bs, self.input_dim))
                    z = z.to(self.device)
                    x_fake = self.generator(z)
                    d_x_fake = self.discriminator(x_fake, 
                                                  growing_step=growing_step)
                    d_x_fake = d_x_fake.mean()
                    d_x_fake.backward()
                    
                    gradient_penalty = self.calc_gradient_panalty(x_real,
                                                                  x_fake, 
                                                                  growing_step)
                    ((gradient_penelty*self.lambda_).mean()).backward()             
                    opt_d.step()

                    # update generator
                    for p in self.discriminator.parameters():
                        p.requires_grad = False
                    for p in self.generator.parameters():
                        p.requires_grad = True
                    
                    self.generator.zero_grad()
                    z = self.z_sampler.sample((bs, self.input_dim))
                    z = z.to(self.device)
                    x_fake = self.generator(z)
                    _d_x_fake = self.discriminator(x_fake,
                                                   growing_step=growing_step)
                    _d_x_fake = -_d_x_fake.mean()
                    _d_x_fake.backward()
                    opt_g.step()

                    print('%.1[s]' % time.time() - start,
                          'WD: %.3f' % -d_x_real.item()+d_x_fake.item(),
                          'GP: %.3f' % ((gradient_penelty*self.lambda_).mean()).item(),
                          'Loss_g: %.3f' % _d_x_fake.item(), 
                          end='\r')

                    if epoch % save_steps == 0:
                        torch.save(self.generator.state_dict(),
                                   os.path.join(current_logdir, 'generator_%d.pth' % epoch))
                        torch.save(self.discriminator.state_dict(),
                                   os.path.join(current_logdir, 'discriminator_%d.pth' % epoch))

                    if epoch % visualize_steps == 0:
                        x_fake = self.generator(fixed_z).detach()
                        self.visualize(os.path.join(current_logdir, 'result_%d.png' % epoch),
                                       x_fake)

    def calc_gradient_panalty(self, x_real, 
                              x_fake, 
                              growing_step):
        bs = x_real.shape[0]
        alpha = torch.rand(bs, 1, 1, 1)
        alpha = alpha.to(self.device)

        interpolates = alpha*d_x_real + (1.-alpha)*d_x_fake
        d_x_inter = self.discriminator(interpolates, 
                                       growing_step=growing_step)

        gradients = torch.autograd.grad(outputs=d_x_inter,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(d_x_inter.size()).to(self.device),
                                        create_graph=True, 
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2

    def visualize(self, dst_path, x):
        n = int(np.sqrt(len(x)))
        x = x[:n**2]
        x = np.transpose(x, (0, 2, 3, 1))
        h, w, c = x.shape[1:]
        x = x.reshape(n, n, *x.shape[1:])
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(n*h, n*w, c)
        if c == 1:
            x = np.squeeze(x, -1)
        x = (x + 1) / 2 * 255
        x = x.numpy().astype('uint8')
        image = Image.fromarray(x)
        image.save(dst_path)

    def generate(self):
        pass
