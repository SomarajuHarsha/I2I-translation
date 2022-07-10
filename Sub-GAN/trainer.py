"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, StyleGenerator, StyleDiscriminator
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class SubGAN_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(SubGAN_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.enc_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.enc_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        # self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        # self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.device = hyperparameters['device']
        
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        self.style_gen_a = StyleGenerator(hyperparameters['gen']) # generator to generate domain a to b style
        self.style_gen_b = StyleGenerator(hyperparameters['gen']) # generator to generate domain b to a style
        self.style_dis_a = StyleDiscriminator(self.style_dim, hyperparameters['dis']) # discriminator for domain a style
        self.style_dis_b = StyleDiscriminator(self.style_dim, hyperparameters['dis']) # discriminator for domain b style

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).to(self.device)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        gen_params = list(self.style_gen_a.parameters()) + list(self.style_gen_b.parameters())
        dis_params = list(self.style_dis_a.parameters()) + list(self.style_dis_b.parameters())
        enc_params = list(self.enc_a.parameters()) + list(self.enc_b.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.enc_opt = torch.optim.Adam([p for p in enc_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.enc_scheduler = get_scheduler(self.enc_opt, hyperparameters)
        
        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.style_dis_a.apply(weights_init('gaussian'))
        self.style_dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        # s_a = Variable(self.s_a)
        # s_b = Variable(self.s_b)
        c_a, s_a = self.enc_a.encode(x_a)
        c_b, s_b = self.enc_b.encode(x_b)
        s_ab = self.style_gen_a(s_a)
        s_ba = self.style_gen_a(s_b)
        x_ba = self.enc_a.decode(c_b, s_ba)
        x_ab = self.enc_b.decode(c_a, s_ab)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))

        # encode
        c_a, s_a_prime = self.enc_a.encode(x_a)
        c_b, s_b_prime = self.enc_b.encode(x_b)

        # generate style
        s_ab = self.style_gen_a(s_a_prime)
        s_ba = self.style_gen_a(s_b_prime)

        # decode (within domain)
        x_a_recon = self.enc_a.decode(c_a, s_a_prime)
        x_b_recon = self.enc_b.decode(c_b, s_b_prime)

        # decode (cross domain)
        x_ba = self.enc_a.decode(c_b, s_a_prime) # u
        x_ab = self.enc_b.decode(c_a, s_b_prime) # v

        # encode again (within domain)
        c_a_recon, s_a_recon = self.enc_a.encode(x_a_recon)
        c_b_recon, s_b_recon = self.enc_b.encode(x_b_recon)

        # encode again (cross domain)
        c_b_crosscon, s_a_crosscon = self.enc_a.encode(x_ba)
        c_a_crosscon, s_b_crosscon = self.enc_b.encode(x_ab)

        # decode again (cross domain)
        x_a_crosscon = self.enc_a.decode(c_a_crosscon, s_a_crosscon)
        x_b_crosscon = self.enc_b.decode(c_b_crosscon, s_b_crosscon)

        # decode again (if needed)
        x_aba = self.enc_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.enc_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # image reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        # # latent reconstruction loss - after gan generation
        # self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        # self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)

        # latent reconstruction loss - without gan generation
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b_prime)

        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        # self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # cross cycle consistency loss
        self.loss_cross_cycle = self.recon_criterion(x_a, x_a_crosscon) + self.recon_criterion(x_b, x_b_crosscon)

        # GAN loss
        # self.loss_enc_adv_a = self.dis_a.calc_gen_loss(x_ba)
        # self.loss_enc_adv_b = self.dis_b.calc_gen_loss(x_ab)
        self.loss_gen_adv_a = self.style_dis_a.calc_gen_loss(s_ba)
        self.loss_gen_adv_b = self.style_dis_a.calc_gen_loss(s_ab)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['cross_cyc_w'] * self.loss_cross_cycle + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.device)
        target_vgg = vgg_preprocess(target, self.device)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.enc_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.enc_b.encode(x_b[i].unsqueeze(0))
            s_ab = self.style_gen_a(s_a_fake)
            s_ba = self.style_gen_b(s_b_fake)
            x_a_recon.append(self.enc_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.enc_b.decode(c_b, s_b_fake))
            x_ba1.append(self.enc_a.decode(c_b, s_ba))
            x_ba2.append(self.enc_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.enc_b.decode(c_a, s_ab))
            x_ab2.append(self.enc_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))
        # encode
        c_a, s_a = self.enc_a.encode(x_a)
        c_b, s_b = self.enc_b.encode(x_b)
        # gen
        s_ab = self.style_gen_a(s_a)
        s_ba = self.style_gen_b(s_b)
        # gen back
        s_bab = self.style_gen_a(s_ba)
        s_aba = self.style_gen_b(s_ab)
        # # decode (cross domain)
        # x_ba = self.enc_a.decode(c_b, s_ba)
        # x_ab = self.enc_b.decode(c_a, s_ab)
        # D loss
        # self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        # self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_a = self.style_dis_a.calc_dis_loss(s_ba.detach(), s_a.detach())
        self.loss_dis_b = self.style_dis_a.calc_dis_loss(s_ab.detach(), s_b.detach() )

        # Cycle adversarial loss
        self.loss_cycle = self.style_dis_a.calc_dis_loss(s_aba.detach(), s_a.detach()) + self.style_dis_b.calc_dis_loss(s_bab.detach(), s_b.detach())

        self.loss_dis_total =   hyperparameters['gan_w'] * self.loss_dis_a + \
                                hyperparameters['gan_w'] * self.loss_dis_b + \
                                hyperparameters['cyc_w'] * self.loss_cycle
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.enc_scheduler is not None:
            self.enc_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load encryptors
        last_model_name = get_model_list(checkpoint_dir, "enc")
        state_dict = torch.load(last_model_name)
        self.enc_a.load_state_dict(state_dict['a'])
        self.enc_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.style_gen_a.load_state_dict(state_dict['a'])
        self.style_gen_b.load_state_dict(state_dict['b'])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.style_dis_a.load_state_dict(state_dict['a'])
        self.style_dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.enc_opt.load_state_dict(state_dict['enc'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.enc_scheduler = get_scheduler(self.enc_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        enc_name = os.path.join(snapshot_dir, 'enc_%08d.pt' % (iterations + 1))
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.enc_a.state_dict(), 'b': self.enc_b.state_dict()}, enc_name)
        torch.save({'a': self.style_gen_a.state_dict(), 'b': self.style_gen_b.state_dict()}, gen_name)
        torch.save({'a': self.style_dis_a.state_dict(), 'b': self.style_dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(), 'enc': self.enc_opt.state_dict()}, opt_name)