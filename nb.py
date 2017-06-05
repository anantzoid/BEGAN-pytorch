
# coding: utf-8

# In[1]:
#changes:lr_update_step, gamma, h
load_model = False#True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import deque

from torchvision import transforms
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.utils.data as data

from PIL import Image
import os
import os.path


lr_update_step = 1000

# In[2]:

b_size = 16
h = 64
nc = 64
epochs = 100
cuda = False
lr = 0.0001
gamma = 0.4

if cuda:
    torch.cuda.set_device(3)

# In[3]:
gen_save_path =   '/misc/vlgscratch2/LecunGroup/anant/began/models/g.pth'
disc_save_path =   '/misc/vlgscratch2/LecunGroup/anant/began/models/d.pth'
data_path = '/misc/vlgscratch2/LecunGroup/anant/began/data/CelebA/'

data_path = 'data/CelebA/'
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    #counter = 0
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
                #counter += 1
                #if counter > 100:
                #    return images

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        print("Found {} images in subfolders of: {}".format(len(imgs), root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

def get_loader(root, split, batch_size, scale_size, num_workers=12, shuffle=True):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)

    if dataset_name in ['CelebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader
data_loader = get_loader(data_path, 'train', b_size, 64)
#data_loader = get_loader('data/CelebA', 'train', b_size, 64)


# In[11]:

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# TODO try putting encoder+decoder as disc if this doesn't work
class Decoder(nn.Module):
    def __init__(self, num_channel):
        super(Decoder, self).__init__()
        self.num_channel = num_channel

        self.l0 = nn.Linear(h, 8*8*num_channel)
        self.l1 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l3 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l5 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l7 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l9 = nn.Conv2d(num_channel, 3, 3, 1, 1)

        
    def forward(self, input):
        x = self.l0(input)
        x = x.view(b_size, self.num_channel,8, 8)
        
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.up1(x)

        x = F.elu(self.l3(x))
        x = F.elu(self.l4(x))
        x = self.up2(x)
        x = F.elu(self.l5(x))
        x = F.elu(self.l6(x))
        x = self.up3(x)
        x = F.elu(self.l7(x))
        x = F.elu(self.l8(x))
        x = F.tanh(self.l9(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_channel):
        super(Encoder, self).__init__()
        self.num_channel = num_channel
        
        '''
        self.l0 = nn.Conv2d(3, num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(num_channel, 2*num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(2*num_channel, 2*num_channel, 3, 2, 1)        

        self.l3 = nn.Conv2d(2*num_channel, 4*num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(4*num_channel, 4*num_channel, 3, 2, 1)        

        self.l5 = nn.Conv2d(4*num_channel, 8*num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(8*num_channel, 8*num_channel, 3, 2, 1)        
        
        self.l7 = nn.Conv2d(8*num_channel, 16*num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(16*num_channel, 16*num_channel, 3, 1, 1)        

        self.l9 = nn.Linear(8*8*16*num_channel, h)
        '''
        self.l0 = nn.Conv2d(3, num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(num_channel, num_channel, 3, 1, 1)
        self.down1 = nn.Conv2d(num_channel, 2*num_channel, 1, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.l3 = nn.Conv2d(2*num_channel, 2*num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(2*num_channel, 2*num_channel, 3, 1, 1)
        self.down2 = nn.Conv2d(2*num_channel, 3*num_channel, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.l5 = nn.Conv2d(3*num_channel, 3*num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(3*num_channel, 3*num_channel, 3, 1, 1)
        self.down3 = nn.Conv2d(3*num_channel, 4*num_channel, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.l7 = nn.Conv2d(4*num_channel, 4*num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(4*num_channel, 4*num_channel, 3, 1, 1)
        self.l9 = nn.Linear(8*8*4*num_channel, h)
        
    def forward(self, input):
        #print "========="
        #print input[0,0,:10,:10]
        x = self.l0(input)
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = self.down1(x)
        x = self.pool1(x)
        
        x = F.elu(self.l3(x))
        x = F.elu(self.l4(x))
        x = self.pool2(self.down2(x))

        x = F.elu(self.l5(x))
        x = F.elu(self.l6(x))
        x = self.pool3(self.down3(x))

        x = F.elu(self.l7(x))
        x = F.elu(self.l8(x))
        x = x.view(b_size, 8*8*4*self.num_channel)
        x = self.l9(x)
        '''
        x = F.elu(self.l0(input))
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        
        x = F.elu(self.l3(x))
        x = F.elu(self.l4(x))
        x = F.elu(self.l5(x))
        x = F.elu(self.l6(x))
        x = F.elu(self.l7(x))
        x = F.elu(self.l8(x))
        x = x.view(b_size, 8*8*16*self.num_channel)
        x = self.l9(x)
        #print "*********"
        #print x[0,:10]
        '''

        return x
    
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.enc = Encoder(nc)
        self.dec = Decoder(nc)
    def forward(self, input):
        return self.dec(self.enc(input))


# In[33]:

disc = Discriminator(nc)
gen = Decoder(nc)
#disc.apply(weights_init)
#gen.apply(weights_init)

if cuda:
    disc.cuda()
    gen.cuda()

if load_model:
    gen.load_state_dict(torch.load(gen_save_path))     
    disc.load_state_dict(torch.load(disc_save_path))     

#decay LR by 0.5 when M stalls
g_opti = torch.optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=lr)
d_opti = torch.optim.Adam(disc.parameters(), betas=(0.5, 0.999), lr=lr)

def criterion(x1, x2):
    return torch.mean(torch.abs(x1-x2))

k = 0
lambda_k = 0.001
z = Variable(torch.FloatTensor(b_size, h))
fixed_z = Variable(torch.FloatTensor(b_size, h))
fixed_x = None
if cuda:
    z = z.cuda()
    fixed_z = fixed_z.cuda()
fixed_z.data.normal_(-1, 1)    
#labels = Variable(torch.FloatTensor(b_size))
#real_label, fake_label = 1, 0
break_step = 100
global_step = 0
convergence_history = []
prev_measure = 1
measure_history = deque([0]*lr_update_step, lr_update_step)

class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # this won't still solve the problem
        # which means gradient will not flow through target
        # _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument `sizeAverage=False`
    """
    pass

criterion = L1Loss()
if cuda:
    criterion.cuda()


for i in range(epochs):
    for _, data in enumerate(data_loader):
        data = Variable(data[0])
        if data.size(0) != b_size:
            continue

        if cuda:
            data = data.cuda()
        if fixed_x is None:
            fixed_x = data
        z.data.normal_(-1, 1)
        #gen.zero_grad()
        disc.zero_grad()

        outputs_d_x = disc(data)
        gen_z = gen(z)
        outputs_d_z = disc(gen_z.detach())
        
        real_loss_d = torch.mean(torch.abs(outputs_d_x - data))
        fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))

        lossD = real_loss_d - k * fake_loss_d
        lossD.backward()
        d_opti.step()
       
        gen.zero_grad()
        gen_z = gen(z)
        outputs_g_z = disc(gen_z)
        lossG = torch.mean(torch.abs(outputs_g_z - gen_z))

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

        balance = (gamma*real_loss_d - fake_loss_d).data[0]
        k += lambda_k * balance
        #k = min(max(0, k), 1)
        k = max(min(1, k), 0)
       
        convg_measure = real_loss_d.data[0] + np.abs(balance) 
        #measure_history.append(convg_measure)
        if global_step%1 == 0:
            print "Step: %d, Loss D: %.9f, fake_loss: %.9f, Loss G: %.9f, k: %f, M: %.9f, lr:%.9f"% (global_step,
                                                lossD.data[0], fake_loss_d.data[0], lossG.data[0], k, convg_measure, lr)
            sample = gen(fixed_z)
            vutils.save_image(sample.data, '/misc/vlgscratch2/LecunGroup/anant/began/samples/%d_gen.png'%global_step)
            recon = disc(fixed_x)
            vutils.save_image(recon.data, '/misc/vlgscratch2/LecunGroup/anant/began/samples/%d_recon.png'%global_step)
           
        
        new_lr = lr* 0.95 ** (global_step//lr_update_step)
        # TODO try just updating param groups if this doesn't work
        g_opti = torch.optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=new_lr)
        d_opti = torch.optim.Adam(disc.parameters(), betas=(0.5, 0.999), lr=new_lr)
        ''' 
        if global_step % lr_update_step == lr_update_step - 1:
            cur_measure = np.mean(measure_history)
            if cur_measure > prev_measure * 0.9999:
                lr *= 0.5
                g_opti = torch.optim.Adam(gen.parameters(), betas=(0.5, 0.999), lr=lr)
                d_opti = torch.optim.Adam(disc.parameters(), betas=(0.5, 0.999), lr=lr)
            prev_measure = cur_measure
        ''' 

        if global_step%1000 == 0:
            torch.save(gen.state_dict(), gen_save_path) 
            torch.save(disc.state_dict(), disc_save_path) 
    
        #convergence_history.append(convg_measure)
        global_step += 1

# In[ ]:




