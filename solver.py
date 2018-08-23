import torch
import time
import os
import numpy as np
from utils.image import tile_images
from generator import Generator
from discriminator import Discriminator
from data_loader import Dataset


class Solver:
    def __init__(self, input_dim=100, 
                 nb_growing=8,
                 upsampling='subpixel',
                 downsampling='avg_pool',
                 device='cuda',
                 lambda_=10.,
                 norm_eps=1e-3,
                 initial_alpha=0.,
                 delta=None):
        if device == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.input_dim = input_dim
        self.nb_growing = nb_growing
        self.device = device
        self.lambda_ = lambda_
        self.norm_eps = norm_eps
        self.initial_alpha = initial_alpha
        self.delta = delta
        self.generator = Generator(input_dim=input_dim,
                                   nb_growing=nb_growing,
                                   upsampling=upsampling).to(device)
        self.discriminator = Discriminator(nb_growing=nb_growing,
                                           downsampling=downsampling).to(device)
        self.resolutions = [(2**(2+i), 2**(2+i)) for i in range(nb_growing)]
        self.z_sampler = torch.distributions.Uniform(-1., 1.)
        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    def fit(self, image_dir: str,
            nb_epoch: int = 100,
            batch_size: object = 32,
            lr_d: float = 2e-4,
            lr_g: float = 2e-4,
            logdir: str = 'logs',
            save_steps: int = 10,
            visualize_steps: int = 1):

        if isinstance(batch_size, int):
            batch_sizes = [batch_size] * self.nb_growing
        else:
            batch_sizes = batch_size
        assert len(batch_sizes) == self.nb_growing

        for growing_step in range(self.nb_growing):
            # prepare components.
            fixed_z = self.z_sampler.sample((batch_sizes[growing_step], self.input_dim))
            fixed_z = fixed_z.to(self.device)

            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr_d,
                                     betas=(0.5, 0.999))
            opt_g = torch.optim.Adam(self.generator.parameters(), lr_g,
                                     betas=(0.5, 0.999))

            current_logdir = os.path.join(logdir, 'growing_step_%d' % (growing_step+1))
            os.makedirs(current_logdir, exist_ok=True)

            dataset = Dataset(target_size=self.resolutions[growing_step],
                              is_flip=True)
            data_loader = dataset.flow_from_directory(image_dir, 
                                                      batch_size=batch_sizes[growing_step])
            print('\n'+('='*20),
                  '\nGrowing Step %d / %d' % (growing_step+1, self.nb_growing),
                  '\n'+('='*20))
            self.discriminator.train()
            self.generator.train()

            alpha = float(self.initial_alpha)
            delta = self.delta if self.delta is not None \
                else 5./nb_epoch/(len(dataset)//batch_sizes[growing_step])

            for epoch in range(1, nb_epoch+1):
                print('\nEpoch %d / %d' % (epoch, nb_epoch))
                start = time.time()
                for iter_, x in enumerate(data_loader):
                    # update discriminator
                    opt_d.zero_grad()
                    x_real = x.to(self.device)
                    bs = x_real.shape[0]

                    if bs != batch_sizes[growing_step]:
                        continue

                    d_x_real = self.discriminator(x_real, 
                                                  growing_step=growing_step,
                                                  alpha=alpha)
                    d_norm = (d_x_real**2).mean()
                    (d_norm*self.norm_eps).backward(retain_graph=True)

                    d_x_real = -d_x_real.mean()
                    d_x_real.backward()

                    z = self.z_sampler.sample((bs, self.input_dim))
                    z = z.to(self.device)
                    x_fake = self.generator(z,
                                            growing_step=growing_step,
                                            alpha=alpha)
                    d_x_fake = self.discriminator(x_fake, 
                                                  growing_step=growing_step,
                                                  alpha=alpha)
                    d_x_fake = d_x_fake.mean()
                    d_x_fake.backward()
                    
                    gradient_penalty = self.calc_gradient_penalty(x_real,
                                                                  x_fake,
                                                                  growing_step,
                                                                  alpha)
                    ((gradient_penalty*self.lambda_).mean()).backward()
                    opt_d.step()

                    # update generator
                    opt_g.zero_grad()
                    z = self.z_sampler.sample((bs, self.input_dim))
                    z = z.to(self.device)
                    x_fake = self.generator(z,
                                            growing_step=growing_step,
                                            alpha=alpha)
                    _d_x_fake = self.discriminator(x_fake,
                                                   growing_step=growing_step,
                                                   alpha=alpha)
                    _d_x_fake = -_d_x_fake.mean()
                    _d_x_fake.backward()
                    opt_g.step()

                    print('%d / %d' % (iter_, len(dataset)//batch_sizes[growing_step]),
                          '%.1f[s]' % (time.time() - start),
                          'WD: %.3f' % (-d_x_real.item()+d_x_fake.item()),
                          'GP: %.3f' % (gradient_penalty.mean()).item(),
                          'Loss_g: %.3f' % _d_x_fake.item(), 
                          end='\r')

                    alpha += delta
                    alpha = np.minimum(alpha, 1.)
                    alpha = float(alpha)

                if epoch % save_steps == 0:
                    torch.save(self.generator.state_dict(),
                               os.path.join(current_logdir, 'generator_%d.pth' % epoch))
                    torch.save(self.discriminator.state_dict(),
                               os.path.join(current_logdir, 'discriminator_%d.pth' % epoch))

                if epoch % visualize_steps == 0:
                    x_fake = self.generator(fixed_z,
                                            growing_step=growing_step).detach()
                    tile_images(os.path.join(current_logdir, 'result_%d.png' % epoch),
                                x_fake)

    def calc_gradient_penalty(self, x_real,
                              x_fake,
                              growing_step,
                              alpha):
        bs = x_real.shape[0]
        phi = torch.rand(bs, 1, 1, 1)
        phi = phi.to(self.device)

        interpolates = phi*x_real + (1.-phi)*x_fake
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        interpolates = interpolates.to(self.device)
        x_inter = self.discriminator(interpolates,
                                     growing_step=growing_step,
                                     alpha=alpha)

        gradients = torch.autograd.grad(outputs=x_inter,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(x_inter.size()).to(self.device),
                                        create_graph=True, 
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def init_weights(self, m):
        is_init = isinstance(m, torch.nn.Conv2d) \
                or isinstance(m, torch.nn.ConvTranspose2d) \
                or isinstance(m, torch.nn.Linear)
        if is_init:
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
