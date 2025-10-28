import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import itertools
import torch
import torchvision
import torchvision.transforms as transforms
import functools
import torch.optim as optim
from torch.autograd import Variable
from comput_loss import cyclegan_loss, compute_loss
from torch.utils.data import Dataset, DataLoader
from dataset import dataloader
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time
import math
from statistics import mean
from tensorboardX import SummaryWriter
from Network import generator, ResNet, UnetDiscriminotor, UnetGenerator
from args import args
import os


is_cuda = torch.cuda.is_available()
writer = SummaryWriter()
# not use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def saveD(dir_chck, GB_A, D_A, D_B, optimG, optimD, epoch):
    if not os.path.exists(dir_chck):
        os.makedirs(dir_chck)

    torch.save({'GB_A': GB_A.state_dict(),
                'D_A': D_A.state_dict(), 'D_B': D_B.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               '%s/modelD_epoch%04d.pth' % (dir_chck, epoch))

def loadD(dir_chck, GB_A, D_A=[], D_B=[], optimG=[], optimD=[], epoch=[], mode='train'):
    if not epoch:
        ckpt = os.listdir(dir_chck)
        ckpt.sort()
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

    dict_net = torch.load('%s/modelD_epoch%04d.pth' % (dir_chck, epoch))

    print('Loaded %dth network' % epoch)

    if mode == 'train':

        GB_A.load_state_dict(dict_net['GB_A'])
        D_A.load_state_dict(dict_net['D_A'])
        D_B.load_state_dict(dict_net['D_B'])
        optimG.load_state_dict(dict_net['optimG'])
        optimD.load_state_dict(dict_net['optimD'])

        return GB_A, D_A, D_B, optimG, optimD, epoch

    elif mode == 'test':

        GB_A.load_state_dict(dict_net['GB_A'])

        return  GB_A, epoch



def main():
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # if args.gpu is not None:
    #     if not len(args.gpu) > torch.cuda.device_count():
    #         ngpus_per_node = len(args.gpu)
    #     else:
    #         print("We will use all available GPUs")
    #         ngpus_per_node = torch.cuda.device_count()
    # else:
    #     ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)



    GB_A = generator(1,1,64,6).cuda(args.gpu)
    D_A = UnetDiscriminotor(num_in_ch=1, num_feat=64, skip_connection=True).cuda(args.gpu)
    D_B = UnetDiscriminotor(num_in_ch = 1, num_feat = 64, skip_connection = True).cuda(args.gpu)

    optimGB = torch.optim.Adam(itertools.chain(GB_A.parameters()), lr=args.lr)
    optimD = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=args.lr)




    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler_GB = optim.lr_scheduler.StepLR(optimGB, step_size=args.lr_step, gamma=0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimD, step_size=args.lr_step, gamma=0.9)


    # st_epoch = 0

    # if args.pre_train == 'on':
    #     GB_A, DA, DB, optimGB, optimD, st_epoch = \
    #         loadD(args.dir_chck, GB_A, D_A, D_B, optimGB, optimD, epoch=args.st_eopch,
    #               mode=args.mode)



    ## setup tensorboard

    trainloader = dataloader(batch_size=args.batch_size, num_workers=args.workers)

    num_train = len(trainloader.dataset)
    num_batch_train = int((num_train / args.batch_size) + ((num_train % args.batch_size) != 0))
    L1_loss = nn.L1Loss().to(device)
    st_epoch = 10

    loss_total_train = []

    for ep in range(st_epoch, args.epochs):
        total_loss = 0

        for idx, data in enumerate(trainloader):

            if is_cuda and args.gpu is not None:
                gt_im_3 = data[0].cuda(args.gpu, non_blocking=True)
                Mosiacimg_4_3 = data[1].cuda(args.gpu, non_blocking=True)
                Mosiacimg_8_3 = data[2].cuda(args.gpu, non_blocking=True)
                Mosiacimg_16_3 = data[3].cuda(args.gpu, non_blocking=True)

                gt_img = gt_im_3[:, 0, :, :].unsqueeze(dim=1)
                Mosiacimg_4 = Mosiacimg_4_3[:, 0, :, :].unsqueeze(dim=1)
                Mosiacimg_8 = Mosiacimg_8_3[:, 0, :, :].unsqueeze(dim=1)
                Mosiacimg_16 = Mosiacimg_16_3[:, 0, :, :].unsqueeze(dim=1)

            optimGB.zero_grad()
            optimD.zero_grad()

            pred_16 = GB_A(gt_img)

            torchvision.utils.save_image(gt_img, args.trainpath + str(ep) + "_gt.png")
            torchvision.utils.save_image(Mosiacimg_16, args.trainpath + str(ep) + "_Mosiacimg_16.png")

            torchvision.utils.save_image(pred_16, args.trainpath + str(ep) + "_pred_16.png")




            loss_total =  L1_loss(pred_16, Mosiacimg_16)

            loss_total.backward(retain_graph=True)
            # optimGA2.step()
            optimGB.step()



            loss_total_train += [loss_total.item()]

            # print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
            #       'loss_gt: %.4f loss_64: %.4f  loss_16: %.4f  loss_G: %.4f loss_D: %.4f'  % (ep, idx, num_batch_train,mean(loss_gt_train), mean(loss_64_train),mean(loss_16_train),mean(loss_G_train), mean(loss_D_train)))
            print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                  'loss_total: %.4f '
                  % (ep, idx, num_batch_train, mean(loss_total_train)))

            saveD(args.dir_chck, GB_A, D_A, D_B, optimGB, optimD, ep)

        # if torch.cuda.current_device() == 0:
        #     print("GPU{} Total_loss:".format(torch.cuda.current_device()), loss)
        scheduler_GB.step()
        scheduler_D.step()


        writer.add_scalars("loss", {"train": mean(loss_total_train)}, ep + 1)
        writer.export_scalars_to_json("./loss.json")
        writer.close()

        if (ep + 1) % args.save_per_epoch == 0:
            saveD(args.dir_chck, GB_A, D_A, D_B, optimGB, optimD, ep)


    print('Finished training')


if __name__ == "__main__":
    main()
