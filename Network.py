import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import time
from transblock import TransformerBlock
from torch.nn.utils import spectral_norm
import functools
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F


# basic arch
class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.decbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.decbr(x)


class ResBlock(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm='inorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []

        # 1st conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        # 2nd conv
        layers += [Padding(padding, padding_mode=padding_mode)]
        layers += [CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=[])]

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)


class CNR1d(nn.Module):
    def __init__(self, nch_in, nch_out, norm='bnorm', relu=0.0, drop=[]):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = []
        layers += [nn.Linear(nch_in, nch_out, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, output_padding=0, bias=True):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # layers = [nn.Upsample(scale_factor=2, mode='bilinear'),
        #           nn.ReflectionPad2d(1),
        #           nn.Conv2d(nch_in , nch_out, kernel_size=3, stride=1, padding=0)]
        #
        # self.deconv = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Padding(nn.Module):
    def __init__(self, padding, padding_mode='zeros', value=0):
        super(Padding, self).__init__()
        if padding_mode == 'reflection':
            self. padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x):
        return self.padding(x)


class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])

        return torch.cat([x2, x1], dim=1)





# transformer model
class generator(nn.Module):
    def __init__(self,in_c,out_c,n_c=64,n_b=6):
        super(generator,self).__init__()
        self.in_c=in_c
        self.out_c=out_c
        self.num_c=n_c
        self.num_b = n_b//2

        self.conv_rec = nn.Sequential(
            nn.Conv2d(1, n_c, 3, 1, 1),
            nn.PReLU()
        )
        self.conv_pred = nn.Sequential(
            nn.Conv2d(1, n_c // 2, 3, 1, 1),
            nn.PReLU()
        )
        self.conv_par = nn.Sequential(
            nn.Conv2d(1, n_c // 4, 3, 1, 1),
            nn.PReLU()
        )
        self.conv_qp = nn.Sequential(
            nn.Conv2d(1, n_c // 4, 3, 1, 1),
            nn.PReLU()
        )
        self.conv_inte = nn.Sequential(
            nn.Conv2d(n_c * 2, n_c, 1, 1, 0),
            nn.PReLU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, 2, 1),
        )

        self.layers = nn.ModuleList([WCDB_block(n_c) for _ in range(self.num_b)])
        self.layers1 = nn.ModuleList([WCDB_block1(n_c) for _ in range(self.num_b)])

        self.conv_last = nn.Sequential(
            nn.Conv2d(n_c, 4, 3, 1, 1)
        )
        self.up = nn.PixelShuffle(2)

    def forward(self,input):
        input = torch.cat((input, input, input, input),dim = 1)
        rec, pred, par, qp_map = input[:, 0:1, :, :], input[:, 1:2, :, :], input[:, 2:3, :, :], input[:, 3:4, :, :]
        #rec = input
        x1 = self.conv_rec(rec)
        x2 = self.conv_pred(pred)
        x3 = self.conv_par(par)
        x4 = self.conv_qp(qp_map)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_inte(x)
        x = self.down(x)

        prior = torch.cat((qp_map, par), dim=1)
        prior = torch.nn.functional.interpolate(prior, scale_factor=0.5)

        for i in range(self.num_b):
            x = self.layers[i](prior, x)
        shallow_fea = x

        for i in range(self.num_b):
            x = self.layers1[i](x)

        out = self.conv_last(x)
        out = self.up(out)
        out = rec + out

        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        out=x * y.expand_as(x)

        return out

class RB_block_trans(nn.Module):
    def __init__(self,ngf=64):
        super(RB_block_trans,self).__init__()

        self.group = 1

        # self.conv = nn.Sequential(
        #     nn.Conv2d(ngf, ngf, 3, 1, 1, groups=self.group),
        #     nn.PReLU(),
        #     nn.Conv2d(ngf, ngf, 3, 1, 1, groups=self.group)
        # )

        self.trans = TransformerBlock(64,2,2.66,True,'WithBias')
        # self.conv_fuse = nn.Conv2d(ngf*2, ngf, 1, 1, 0)
        # self.seBlock=SEBlock(ngf)

    def forward(self,input):

        x = input

        # x1 = self.conv(x)
        x1 = self.trans(x)
        out = x1

        return out

class RB_block_conv(nn.Module):
    def __init__(self,ngf=64):
        super(RB_block_conv,self).__init__()

        self.group = 1

        self.conv = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, groups=self.group),
            nn.PReLU(),
            nn.Conv2d(ngf, ngf, 3, 1, 1, groups=self.group)
        )

        # self.trans = TransformerBlock(64,2,2.66,True,'WithBias')
        # self.conv_fuse = nn.Conv2d(ngf*2, ngf, 1, 1, 0)
        # self.seBlock=SEBlock(ngf)

    def forward(self,input):

        x = input

        x1 = self.conv(x)
        # x1 = self.trans(x)
        out = x1 + input

        return out


class prior_att(nn.Module):
    def __init__(self, in_c=64):
        super(prior_att, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(4, in_c, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(in_c, in_c, 3,1,1),
            nn.Sigmoid()
        )

        self.ratio = 4
        self.linear1 = nn.Sequential(
            nn.Conv2d(in_c, int(in_c // self.ratio), 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(int(in_c // self.ratio), int(in_c // self.ratio), 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(int(in_c // self.ratio), in_c, 1, 1, 0),
            nn.Sigmoid()
        )

        self.linear2 = nn.Sequential(
            nn.Conv2d(in_c, int(in_c // self.ratio), 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(int(in_c // self.ratio), int(in_c // self.ratio), 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(int(in_c // self.ratio), in_c, 1, 1, 0),
            nn.Sigmoid()
        )
        self.linear3 = nn.Sequential(
            nn.Conv2d(1, in_c, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_c, 2, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, prior, fea):

        # Spatial attention
        max_fea = torch.max(fea, dim=1).values.unsqueeze(1)
        avg_fea = torch.mean(fea, dim=1).unsqueeze(1)
        inp = torch.cat((prior, max_fea, avg_fea), dim=1)
        mask = self.conv_block(inp)
        out1 = fea * mask

        # Channel attention

        qp_vector = torch.mean(torch.nn.AdaptiveAvgPool2d(1)(prior[:,0:,:,:]),dim=1).unsqueeze(1) # (b, 1,1,1)
        mean = torch.nn.AdaptiveAvgPool2d(1)(fea)
        # Contrast
        sta_var = (fea - mean) ** 2
        var = torch.sqrt(torch.nn.AdaptiveAvgPool2d(1)(sta_var))
        z = var + mean
        mask1 = self.linear1(z)

        # Quality
        mask2 = self.linear2(mean)

        # weighted
        weights = self.linear3(qp_vector)
        alpha, beta = weights[:,0:1,:,:], weights[:,1:,:,:]
        mask_weighted = alpha * mask1 + beta * mask2
        out2 = fea * mask_weighted

        out = out1 + out2

        return out


class WCDB_block(nn.Module):
    def __init__(self,ngf=64,n_b=4):
        super(WCDB_block,self).__init__()

        self.num_b = n_b
        self.RB_layers=nn.ModuleList([RB_block_conv(ngf) for _ in range(n_b)])
        self.attention = prior_att(ngf)

    def forward(self, prior, input):

        x = input
        for i in range(0, self.num_b):
            x = self.RB_layers[i](x)
        att = self.attention(prior, x)
        out = x + att
        return out

class WCDB_block1(nn.Module):
    def __init__(self,ngf=64,n_b=2):
        super(WCDB_block1,self).__init__()

        self.num_b = n_b
        self.RB_layers=nn.ModuleList([RB_block_trans(ngf) for _ in range(n_b)])
        # self.attention = prior_att(ngf)

    def forward(self, input):

        x = input
        for i in range(0, self.num_b):
            x = self.RB_layers[i](x)
        # att = self.attention(prior, x)
        # out = x + att
        out = x
        return out


# resnet generator
class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)

        self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]

            self.res = nn.Sequential(*res)

        self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)

    def forward(self, x1):


        x = self.enc1(x1)
        x = self.enc2(x)
        x = self.enc3(x)

        if self.nblk:
            x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ConectBlock(nn.Module):
    def __init__(self, input_nc,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=5, padding_type='reflect'):
        super(ConectBlock, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        relu = nn.ReLU(True)
        extra_conv = nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=0, bias=use_bias)
        model = [relu]
        for i in range(n_blocks):
            model += [ResnetBlock(input_nc, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.ReflectionPad2d(1), extra_conv, norm_layer(input_nc)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, innerres=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        elif innerres:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print(x.shape)
        # y = self.model(x)
        # print(y.shape)
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, num_res, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        resnet_block = ConectBlock(ngf * 8, use_dropout=use_dropout, n_blocks=num_res)
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=resnet_block,
                                        norm_layer=norm_layer, innerres=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                            norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                        norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                        norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                        norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


### unet discriminator
class UnetDiscriminotor(nn.Module):
    def __init__(self, num_in_ch, num_feat = 64, skip_connection = True):
        super(UnetDiscriminotor, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        self.conv0 = norm(nn.Conv2d(num_in_ch, num_feat, kernel_size = 3, stride = 1, padding = 1))

        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, kernel_size=4, stride=2, padding=1, bias = False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=4, stride=2, padding=1, bias = False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, kernel_size=4, stride=2, padding=1, bias = False))

        #upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=False))

        #extra convolution
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv9 = norm(nn.Conv2d(num_feat, 1, kernel_size=3, stride=1, padding=1))

    def forward(self,x):

        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        #upsamle
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0


        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out =  F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)

        out = self.conv9(out)

        return out






def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


if __name__ == "__main__":
    img = torch.zeros(32,1,144,144)
    #qp = torch.zeros(32, 1, 144, 144)
    G = generator(4,1,64,6)
    # img = torchvision.transforms.ToTensor()(img)
    tic = time.time()
    #img = torch.cat((img,img,img,qp), dim=1)
    print(img.size())
    out = G(torch.cat((img,img,img,img),dim=1))
    print(out.size())
    print(sum(p.numel() for p in G.parameters()))
    toc = time.time()
    print(toc-tic)