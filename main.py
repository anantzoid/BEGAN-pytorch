
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import deque

import torchvision.utils as vutils

import os
import os.path

from dataloader import *
from models import *
import argparse

import random
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--ngpu', default=1, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--b_size', default=16, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_update_step', default=3000, type=int)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--lr_lower_boundary', default=2e-6, type=float)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--lambda_k', default=0.001, type=float)
parser.add_argument('--k', default=0, type=float)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--model_name', default='test2')
parser.add_argument('--base_path', default='/misc/vlgscratch2/LecunGroup/anant/began/')
parser.add_argument('--data_path', default='data/64_crop')
parser.add_argument('--load_step', default=0, type=int)
parser.add_argument('--print_step', default=100, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--l_type', default=1, type=int)
parser.add_argument('--tanh', default=1, type=int)
parser.add_argument('--manualSeed', default=5451, type=int)
parser.add_argument('--train', default=1, type=int)
opt = parser.parse_args()


print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    opt.cuda = True
    torch.cuda.set_device(opt.gpuid)
    torch.cuda.manual_seed_all(opt.manualSeed)


class BEGAN():
    def __init__(self):
        self.global_step = opt.load_step
        self.prepare_paths()
        self.data_loader = get_loader(self.data_path, opt.b_size, opt.scale_size, opt.num_workers)

        self.build_model() 

        self.z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z.data.uniform_(-1, 1)    
        self.fixed_x = None
        
        self.criterion = L1Loss()

        if opt.cuda:
            self.set_cuda()

    def set_cuda(self):
        self.disc.cuda()
        self.gen.cuda()
        self.z = self.z.cuda()
        self.fixed_z = self.fixed_z.cuda()
        self.criterion.cuda()

    def write_config(self, step):
        f = open(os.path.join(opt.base_path, 'experiments/%s/params/%d.cfg'%(opt.model_name, step)), 'w')
        print >>f, vars(opt)
        f.close()
 
    def prepare_paths(self):
        self.data_path = os.path.join(opt.base_path, opt.data_path)
        self.gen_save_path = os.path.join(opt.base_path, 'experiments/%s/models'%opt.model_name)
        self.disc_save_path = os.path.join(opt.base_path, 'experiments/%s/models'%opt.model_name)
        self.sample_dir = os.path.join(opt.base_path,  'experiments/%s/samples'%opt.model_name)
        param_dir =  os.path.join(opt.base_path,  'experiments/%s/params'%opt.model_name)

        for path in [self.gen_save_path, self.disc_save_path, self.sample_dir, param_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
        print("Generated samples saved in %s"%self.sample_dir)
    
    def build_model(self):
        self.disc = Discriminator(opt)
        self.gen = Decoder(opt)
        print self.disc
        print "===================="
        print self.gen
        #disc.apply(weights_init)
        #gen.apply(weights_init)

        if opt.load_step > 0:
            self.load_models(opt.load_step)

    def generate(self, sample, recon, step, nrow=8):
        #sample = self.gen(fake)
        #print sample.size()
        #return
        #sample = sample.data.cpu().mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        #from PIL import Image
        #print type(sample)
        #im = Image.fromarray(sample.astype('uint8'))
        #im.save('128.png')
        vutils.save_image(sample.data, '%s/%s_%s_gen.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
        #f = open('%s/%s_gen.mat'%(self.sample_dir, opt.model_name), 'w')
        #np.save(f, sample.data.cpu().numpy())
        #recon = self.disc(self.fixed_x)
        if recon is not None:
            vutils.save_image(recon.data, '%s/%s_%s_disc.png'%(self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)

    def save_models(self, step):
        torch.save(self.gen.state_dict(), os.path.join(self.gen_save_path, 'gen_%d.pth'%step)) 
        torch.save(self.disc.state_dict(), os.path.join(self.disc_save_path, 'disc_%d.pth'%step)) 
        self.write_config(step)

    def load_models(self, step):
        self.gen.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'gen_%d.pth'%step)))     
        self.disc.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'disc_%d.pth'%step)))     

    def compute_disc_loss(self, outputs_d_x, data, outputs_d_z, gen_z):
        if opt.l_type == 1:
            real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
            fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
        else:
            real_loss_d = self.criterion(outputs_d_x, data)
            fake_loss_d = self.criterion(outputs_d_z , gen_z.detach())
        return (real_loss_d, fake_loss_d)
            
    def compute_gen_loss(self, outputs_g_z, gen_z):
        if opt.l_type == 1:
            return torch.mean(torch.abs(outputs_g_z - gen_z))
        else:
            return self.criterion(outputs_g_z, gen_z)

    def train(self):
        g_opti = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        d_opti = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        measure_history = deque([0]*opt.lr_update_step, opt.lr_update_step)

        convergence_history = []
        prev_measure = 1

        lr = opt.lr

        for i in range(opt.epochs):
            for _, data in enumerate(self.data_loader):
                data = Variable(data)
                if data.size(0) != opt.b_size:
                    print data.size(0)
                    print opt.b_size
                    continue

                if opt.cuda:
                    data = data.cuda()
                if self.fixed_x is None:
                    self.fixed_x = data
                #self.gen.zero_grad()
                self.disc.zero_grad()

                self.z.data.uniform_(-1, 1)
                gen_z = self.gen(self.z)
                outputs_d_z = self.disc(gen_z.detach())
                outputs_d_x = self.disc(data)
               
                real_loss_d, fake_loss_d = self.compute_disc_loss(outputs_d_x, data, outputs_d_z, gen_z)

                lossD = real_loss_d - opt.k * fake_loss_d
                lossD.backward()
                d_opti.step()
            
                self.gen.zero_grad()
                #self.disc.zero_grad()
                gen_z = self.gen(self.z)
                outputs_g_z = self.disc(gen_z)
                lossG = self.compute_gen_loss(outputs_g_z, gen_z)

                #real_loss_d = criterion(outputs_d_x, data)
                #fake_loss_d = criterion(outputs_d_z, gen_z.detach())
                    
                '''
                print "Fake LOSS:............."
                print outputs_d_z[0,0,:10,:10]
                print ".............."
                print gen_z[0,0,:10,:10]
                print "fake_loss_d:==>",fake_loss_d
                '''


                #lossG = criterion(gen_z, outputs_g_z)
                #loss = lossD + lossG
                lossG.backward()


                g_opti.step()

                balance = (opt.gamma*real_loss_d - fake_loss_d).data[0]
                opt.k += opt.lambda_k * balance
                #k = min(max(0, k), 1)
                opt.k = max(min(1, opt.k), 0)
            
                convg_measure = real_loss_d.data[0] + np.abs(balance) 
                measure_history.append(convg_measure)
                if self.global_step%opt.print_step == 0:
                    print "Step: %d, Epochs: %d, Loss D: %.9f, real_loss: %.9f, fake_loss: %.9f, Loss G: %.9f, k: %f, M: %.9f, lr:%.9f"% (self.global_step, i, 
                                                        lossD.data[0], real_loss_d.data[0], fake_loss_d.data[0], lossG.data[0], opt.k, convg_measure, lr)
                    self.generate(gen_z, outputs_d_x, self.global_step)
               
                if opt.lr_update_type == 1:
                    lr = opt.lr* 0.95 ** (self.global_step//opt.lr_update_step)
                elif opt.lr_update_type == 2:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step -1 :
                        lr *= 0.5
                elif opt.lr_update_type == 3:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step -1 :
                        lr = min(lr*0.5, opt.lr_lower_boundary)
                else:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        cur_measure = np.mean(measure_history)
                        if cur_measure > prev_measure * 0.9999:
                            lr = min(lr*0.5, opt.lr_lower_boundary)
                        prev_measure = cur_measure
     

                for p in g_opti.param_groups + d_opti.param_groups:
                    p['lr'] = lr
                # g_opti = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=lr)
                # d_opti = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=lr)
                ''' 
                if self.global_step % lr_update_step == lr_update_step - 1:
                    cur_measure = np.mean(measure_history)
                    if cur_measure > prev_measure * 0.9999:
                        lr *= 0.5
                        g_opti = torch.optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=lr)
                        d_opti = torch.optim.Adam(disc.parameters(), betas=(0.5, 0.999), lr=lr)
                    prev_measure = cur_measure
                ''' 

                if self.global_step%1000 == 0:
                    self.save_models(self.global_step)
            
                #convergence_history.append(convg_measure)
                self.global_step += 1

def generative_experiments(obj):
    z = []
    for inter in range(10):
        z0 = np.random.uniform(-1,1,opt.h)
        z10 = np.random.uniform(-1,1,opt.h)
        def slerp(val, low, high):
            omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
            so = np.sin(omega)
            if so == 0:
                return (1.0-val) * low + val * high # L'Hopital's rule/LERP
            return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high 

        z.append(z0)
        for i in range(1, 9):
            z.append(slerp(i*0.1, z0, z10))
        z.append(z10.reshape(1, opt.h)) 
    z = [_.reshape(1, opt.h) for _ in z]
    z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
    print z_var.size()
    if opt.cuda:
        z_var = z_var.cuda()
    gen_z = obj.gen(z_var)
    obj.generate(gen_z, None, 'gen_1014_slerp_%d'%opt.load_step, 10)

    '''
    # Noise arithmetic 
    for i in range(5):
        sum_z = z[i] + z
        z_var = Variable(torch.from_numpy(np.concatenate(z, 0)).float())
        print z_var.size()
        if opt.cuda:
            z_var = z_var.cuda()
        gen_z = obj.gen(z_var)
        obj.generate(gen_z, None, 'gen_1014_slerp_%d'%i)
    '''
           
            
        

if __name__ == "__main__":
    obj = BEGAN()
    if opt.train:
        obj.train()
    else:
        generative_experiments(obj)
