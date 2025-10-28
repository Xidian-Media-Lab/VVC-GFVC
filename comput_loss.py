import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchvision.models import vgg19
from SSIMLoss import SSIM
import warnings

from dists import *
from vggloss import  *
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from math import exp
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 计算一维的高斯分布向量


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def vgg19_loss(feature_module, loss_func, gt_img, pred_gt):
    F_gt = feature_module(gt_img)
    F_pred = feature_module(pred_gt)

    loss = loss_func(F_gt, F_pred)

    return loss

def get_feature_module(layer_index, device = None):
    vgg = vgg19(pretrained = True, progress = True).features
    vgg.eval()

    for parm in vgg.parameters():
        parm.requires_grad = False

    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return  feature_module

class Perceptualloss(nn.Module):
    def __init__(self, loss_func, layer_indexs = None, device = None):
        super(Perceptualloss, self).__init__()
        self.creation = loss_func
        self.layer_indexs = layer_indexs
        self.device = device

    def forward(self, gt, pred_gt):
        loss = 0
        for index in self.layer_indexs:
            feature_module = get_feature_module(index, self.device)
            loss+=vgg19_loss(feature_module, self.creation, gt, pred_gt)
        return loss

class fftloss(nn.Module):
    def __init__(self):
        super(fftloss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, x, y):
        grad_x_x = torch.abs(x[:,:,:,:-1] - x[:,:,:,1:])
        grad_y_x = torch.abs(x[:,:,:-1,:] - x[:,:,1:,:])
        grad_x_y = torch.abs(y[:,:,:,:-1] - y[:,:,:,1:])
        grad_y_y = torch.abs(y[:,:,:-1,:] - y[:,:,1:,:])
        loss_x = torch.mean(torch.abs(grad_x_y - grad_x_x))
        loss_y = torch.mean(torch.abs(grad_y_y - grad_y_x))
        gradientloss = loss_x + loss_y
        return gradientloss



def cyclegan_loss(gt_img, Mosiacimg_64, pred_real_gt, pred_real_64, pred_fake_64, pred_fake_gt, recon_gt, recon_64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn_Cycle = nn.L1Loss().to(device)  # L1
    fn_GAN = nn.BCEWithLogitsLoss().to(device)

    #loss_D_a
    loss_D_a_real = fn_GAN(pred_real_gt, torch.ones_like(pred_real_gt))
    loss_D_a_fake = fn_GAN(pred_fake_gt, torch.zeros_like(pred_fake_gt))
    loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

    ##loss_D_b
    loss_D_b_real = fn_GAN(pred_real_64, torch.ones_like(pred_real_64))
    loss_D_b_fake = fn_GAN(pred_fake_64, torch.zeros_like(pred_fake_64))
    loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

    loss_D = loss_D_a + loss_D_b

    ##loss identity

    # loss_I_a = fn_Ident(idt_64, gt_img)
    # loss_I_b = fn_Ident(idt_gt, Mosiacimg_64)


    ##loss_G_a , loss_G_b

    loss_G_a2b = fn_GAN(pred_fake_64, torch.ones_like(pred_fake_64))
    loss_G_b2a = fn_GAN(pred_fake_gt, torch.ones_like(pred_fake_gt))

    loss_C_a = fn_Cycle(Mosiacimg_64, recon_64)
    loss_C_b = fn_Cycle(gt_img, recon_gt)

    loss_G = (loss_G_a2b + loss_G_b2a) + \
             (1e-1 * loss_C_a + 1e-1 * loss_C_b)



    return loss_G, loss_D

def distsloss(gt_img,pred_gt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = Vgg19().to(device)
    dists = DISTS().to(device)
    loss_values={} ######
    loss_weights={}
    loss_weights['perceptual_final']=[10,10,10,10,10]
    loss_weights['generator_gan']=1
    scales=[1, 0.5, 0.25, 0.125]
    pyramid = ImagePyramide(scales, 1).to(device)
    #### rd loss optimization
    # rdloss = lamdaloss * bpp_mv + dists  ###
    # loss_values['rdloss'] = rdloss
    pyramide_real= pyramid(gt_img)
    pyramide_generated=pyramid(pred_gt)
    loss_values['dists'] = dists(gt_img, pred_gt, as_loss=True)
    ### Perceptual Loss---Initial
        ### Perceptual Loss---Final
    if sum(loss_weights['perceptual_final']) != 0:
        value_total = 0
        for scale in scales:
            x_vgg = vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = vgg(pyramide_real['prediction_' + str(scale)])

            for i, weight in enumerate(loss_weights['perceptual_final']):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += loss_weights['perceptual_final'][i] * value
            loss_values['perceptual_256FINAL'] = value_total
    return loss_values
    # Gan_loss



def compute_loss(gt_img, pred_gt):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_img, pred_gt \
        = (gt_img.to(device),  pred_gt.to(device))
    # l1
    loss_l1 = nn.L1Loss().to(device)
    loss_fft = fftloss()

    loss_ssim = SSIM()
    loss_ssim = loss_ssim.to(device)
    ssim_gt = (1 - loss_ssim(pred_gt, gt_img))
    # loss_grad = gradient_loss()
    # loss_grad = loss_grad.to(device)
    loss_gt = loss_l1(pred_gt, gt_img) + 0.05 * loss_fft(pred_gt, gt_img)


    #percetural
    gt3 = torch.cat([gt_img, gt_img, gt_img], 1)
    pred_gt3 = torch.cat([pred_gt, pred_gt, pred_gt], 1)
    layer_indexs = [3, 8, 15, 22]
    loss_MSE = nn.MSELoss().to(device)
    creation = Perceptualloss(loss_MSE, layer_indexs, device)
    perceptual_loss = creation(pred_gt3, gt3)

    loss_gt =  loss_gt  + 0.05 * perceptual_loss + ssim_gt
    #loss_total = loss_gt + loss_4 + loss_8 + loss_16 + loss_32,


    return loss_gt


if __name__ == "__main__":
    gt_img = torch.randn(1, 1, 256, 256)
    pred_gt = torch.randn(1, 1, 256, 256)
    Mosiacimg_4 = torch.randn(1, 1, 256, 256)
    pred_4 = torch.randn(1, 1, 256, 256)
    Mosiacimg_16 = torch.randn(1, 1, 256, 256)
    pred_16 = torch.randn(1, 1, 256, 256)
    Mosiacimg_64 = torch.randn(1, 1, 256, 256)
    pred_64 = torch.randn(1, 1, 256, 256)

    l_fft, loss_gt, loss_4, loss_16 = compute_loss(gt_img, Mosiacimg_4, Mosiacimg_16, pred_16, pred_4, pred_gt)

    print(l_fft)
    print(loss_gt)
    print(loss_16)




