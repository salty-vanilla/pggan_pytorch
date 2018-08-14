import torch
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
            lr_g=2e-4):

        self.discriminator.train()
        self.generator.train()
        fixed_z = self.z_sampler.sample((batch_size, self.input_dim))
        for growing_step in range(self.nb_growing):
            dataset = Dataset(target_size=self.resolutions[growing_step])
            data_loader = dataset.flow_from_directory(image_dir, 
                                                      batch_size=batch_size)
            
            opt_d = optim.Adam(self.discriminator.parameters(), lr_d,
                               betas=(0.5, 0.999)) 
            opt_g = optim.Adam(self.generator.parameters(), lr_g,
                               betas=(0.5, 0.999))
            for epoch in range(1, nb_epoch+1):
                for iter_, x in data_loader:
                    bs = x_real.shape[0]

                    # update discriminator
                    for p in self.discriminator.parameters():
                        p.requires_grad = True
                    for p in self.generator.parameters():
                        p.requires_grad = False

                    self.discriminator.zero_grad()
                    x_real = x.to(self.device)
                    d_x_real = self.discriminator(x_real)
                    d_x_real = -d_x_real.mean()
                    d_x_real.backward()

                    z = self.z_sampler.sample((bs, self.input_dim))
                    z = z.to(self.device)
                    x_fake = self.generator(z)
                    d_x_fake = self.discriminator(x_fake)
                    d_x_fake = d_x_fake.mean()
                    d_x_fake.backward()
                    
                    gradient_penalty = self.calc_gradient_panalty(x_real,
                                                                  x_fake)
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
                    d_x_fake = self.discriminator(x_fake)
                    d_x_fake = -d_x_fake.mean()
                    d_x_fake.backward()
                    opt_g.step()

    def calc_gradient_panalty(self, x_real, 
                              x_fake):
        bs = x_real.shape[0]
        alpha = torch.rand(bs, 1, 1, 1)
        alpha = alpha.to(self.device)

        interpolates = alpha*d_x_real + (1.-alpha)*d_x_fake
        d_x_inter = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=d_x_inter,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones(d_x_inter.size()).to(self.device),
                                        create_graph=True, 
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty


        pass


    def generate(self):
        pass
