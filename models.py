import torch.nn as nn
import torch.nn.functional as F

# TODO try putting encoder+decoder as disc if this doesn't work
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.num_channel = opt.nc
        self.b_size = opt.b_size
        self.h = opt.h

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
        self.l9 = nn.Conv2d(self.num_channel, 3, 3, 1, 1)

        
    def forward(self, input):
        x = self.l0(input)
        x = x.view(self.b_size, self.num_channel,8, 8)
        
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
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.num_channel = opt.nc
        self.h = opt.h
        self.b_size = opt.b_size
        
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
        self.l0 = nn.Conv2d(3, self.num_channel, 3, 1, 1)
        self.l1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.l2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channel, 2*self.num_channel, 1, 1, 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.l3 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.l4 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.down2 = nn.Conv2d(2*self.num_channel, 3*self.num_channel, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.l5 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.l6 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.down3 = nn.Conv2d(3*self.num_channel, 4*self.num_channel, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.l7 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
        self.l8 = nn.Conv2d(4*self.num_channel, 4*self.num_channel, 3, 1, 1)
        self.l9 = nn.Linear(8*8*4*self.num_channel, self.h)
        
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
        x = x.view(self.b_size, 8*8*4*self.num_channel)
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
        x = x.view(opt.b_size, 8*8*16*self.num_channel)
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


