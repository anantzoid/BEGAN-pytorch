import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, opt, disc=False):
        super(Decoder, self).__init__()
        self.num_channel = opt.nc
        self.b_size = opt.b_size
        self.h = opt.h
        self.disc = disc
        self.t_act = opt.tanh
        self.scale_size = opt.scale_size

        self.l0 = nn.Linear(self.h, 8*8*self.num_channel)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l5 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.l7 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        if self.scale_size == 128:
            self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
            self.l10 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l9 = nn.Conv2d(self.num_channel, 3, 3, 1, 1)
            

        
    def forward(self, input):
        x = self.l0(input)
        x = x.view(self.b_size, self.num_channel,8, 8)
        
        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.up1(x)

        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.up2(x)
        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.up3(x)
        x = F.elu(self.l7(x), True)
        x = F.elu(self.l8(x), True)
        if self.scale_size == 128:
            x = self.up4(x)
            x = F.elu(self.l10(x))
            x = F.elu(self.l11(x))
        x = self.l9(x)
        #if not self.disc:
        #if self.scale_size != 128:# and self.t_act:
        x = F.tanh(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.num_channel = opt.nc
        self.h = opt.h
        self.b_size = opt.b_size
        self.scale_size = opt.scale_size
        self.l0 = nn.Conv2d(3, self.num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channel, self.num_channel, 1, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.l3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down2 = nn.Conv2d(self.num_channel, 2*self.num_channel, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.l5 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.down3 = nn.Conv2d(2*self.num_channel, 3*self.num_channel, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)
        
        
        if self.scale_size == 64:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l9 = nn.Linear(8*8*3*self.num_channel, 64)
        elif self.scale_size == 128:
            self.l7 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.l8 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.l9 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l11 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
            self.l12 = nn.Linear(8*8*4*self.num_channel, self.h)

        
    def forward(self, input):
        x = F.elu(self.l0(input), True)
        x = F.elu(self.l1(x), True)
        x = F.elu(self.l2(x), True)
        x = self.down1(x)
        x = self.pool1(x)
        
        x = F.elu(self.l3(x), True)
        x = F.elu(self.l4(x), True)
        x = self.pool2(self.down2(x))

        x = F.elu(self.l5(x), True)
        x = F.elu(self.l6(x), True)
        x = self.pool3(self.down3(x))

        if self.scale_size == 64:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = x.view(self.b_size, 8*8*3*self.num_channel)
            x = self.l9(x)
        else:
            x = F.elu(self.l7(x), True)
            x = F.elu(self.l8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.l9(x), True)
            x = F.elu(self.l11(x), True)
            x = x.view(self.b_size, 8*8*4*self.num_channel)
            x = F.elu(self.l12(x), True)

        return x
    
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.enc = Encoder(nc)
        self.dec = Decoder(nc, True)
    def forward(self, input):
        return self.dec(self.enc(input))


def weights_init(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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


