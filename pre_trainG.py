import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import itertools
import torch
import torchvision
import torchvision.transforms as transforms

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
from Network import generator, ResNet, UnetDiscriminotor, init_net
from args import args
import os

is_cuda = torch.cuda.is_available()
writer = SummaryWriter()



# not use

def save(dir_chck, G1A_B, G2A_B, G3A_B, optimG, epoch):
    if not os.path.exists(dir_chck):
        os.makedirs(dir_chck)

    torch.save({'G1A_B': G1A_B.state_dict(), 'G2A_B': G2A_B.state_dict(),
                'G3A_B': G3A_B.state_dict(),
                'optimG': optimG.state_dict()},
               '%s/modelGA_epoch%04d.pth' % (dir_chck, epoch))


def load(dir_chck, G1A_B, G2A_B, G3A_B,optimG=[], epoch=[],
         mode='train'):
    if not epoch:
        ckpt = os.listdir(dir_chck)
        ckpt.sort()
        epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

    dict_net = torch.load('%s/modelGA_epoch%04d.pth' % (dir_chck, epoch))

    print('Loaded %dth network' % epoch)

    if mode == 'train':
        G1A_B.load_state_dict(dict_net['G1A_B'])
        G2A_B.load_state_dict(dict_net['G2A_B'])
        G3A_B.load_state_dict(dict_net['G3A_B'])

        optimG.load_state_dict(dict_net['optimG'])


        return G1A_B, G2A_B, G3A_B,optimG, epoch

    elif mode == 'test':
        G1A_B.load_state_dict(dict_net['G1A_B'])
        G2A_B.load_state_dict(dict_net['G2A_B'])
        G3A_B.load_state_dict(dict_net['G3A_B'])

        return G1A_B, G2A_B, G3A_B


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

    torch.cuda.set_device(args.gpu)

    # G1A_B = ResNet(nch_in=1, nch_out=1, nch_ker=64, norm='inorm', nblk=6).cuda(args.gpu)
    # G2A_B = ResNet(nch_in=1, nch_out=1, nch_ker=64, norm='inorm', nblk=6).cuda(args.gpu)
    # G3A_B = ResNet(nch_in=1, nch_out=1, nch_ker=64, norm='inorm', nblk=6).cuda(args.gpu)

    G1A_B = generator(2, 1, 64, 6).cuda(args.gpu)
    G2A_B = generator(2, 1, 64, 6).cuda(args.gpu)
    G3A_B = generator(2, 1, 64, 6).cuda(args.gpu)
    #G4A_B = generator(1,1,64,6).cuda(args.gpu)
    #G5A_B = generator(1,1,64,6).cuda(args.gpu)


    optimG = torch.optim.Adam(itertools.chain(G1A_B.parameters(), G2A_B.parameters(), G3A_B.parameters()),
                              lr=args.lr)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler_G = optim.lr_scheduler.StepLR(optimG, step_size=args.lr_step, gamma=0.9)


    st_epoch = 0

    if args.pre_train == 'on':
        G1A_B, G2A_B, G3A_B, optimG, st_epoch = \
            load(args.dir_chck, G1A_B, G2A_B, G3A_B, optimG, epoch=1, mode=args.mode)


    ## setup tensorboard

    trainloader = dataloader(batch_size=args.batch_size, num_workers=args.workers)

    num_train = len(trainloader.dataset)
    num_batch_train = int((num_train / args.batch_size) + ((num_train % args.batch_size) != 0))

    loss_gt_train = []
    loss_42_train = []
    loss_32_train = []
    # loss_22_train = []
    loss_total_train = []


    for ep in range(st_epoch+1, args.epochs):
        total_loss = 0

        for idx, data in enumerate(trainloader):

            if is_cuda and args.gpu is not None:
                gt_im_3 = data[0].cuda(args.gpu, non_blocking=True)
                # qp22_img_3 = data[1].cuda(args.gpu, non_blocking=True)
                qp32_img_3 = data[2].cuda(args.gpu, non_blocking=True)
                qp42_img_3 = data[3].cuda(args.gpu, non_blocking=True)
                QP52_img_3 = data[4].cuda(args.gpu, non_blocking=True)
                ref_img_3 = data[5].cuda(args.gpu, non_blocking=True)

                gt_img = gt_im_3[:, 0, :, :].unsqueeze(dim=1)
                # qp22_img = qp22_img_3[:, 0, :, :].unsqueeze(dim=1)
                qp32_img = qp32_img_3[:, 0, :, :].unsqueeze(dim=1)
                qp42_img = qp42_img_3[:, 0, :, :].unsqueeze(dim=1)
                QP52_img = QP52_img_3[:, 0, :, :].unsqueeze(dim=1)
                ref_img = ref_img_3[:, 0, :, :].unsqueeze(dim=1)

            optimG.zero_grad()

            pred_42 = G1A_B(torch.cat((QP52_img, ref_img), dim=1))
            pred_32 = G2A_B(torch.cat((pred_42, ref_img), dim=1))
            pred_gt = G3A_B(torch.cat((pred_32, ref_img), dim=1))
            # pred_gt = G4A_B(torch.cat((pred_22, ref_img), dim=1))

            torchvision.utils.save_image(pred_gt, args.trainpath + str(ep) + "_pred_gt.png")
            torchvision.utils.save_image(pred_42, args.trainpath + str(ep) + "_pred_42.png")
            torchvision.utils.save_image(pred_32, args.trainpath + str(ep) + "_pred_32.png")
            # torchvision.utils.save_image(pred_22, args.trainpath + str(ep) + "_pred_22.png")
            torchvision.utils.save_image(ref_img, args.trainpath + str(ep) + "_ref.png")

            torchvision.utils.save_image(gt_img, args.trainpath + str(ep) + "_gt_img.png")
            # torchvision.utils.save_image(qp22_img, args.trainpath + str(ep) + "_qp22_img.png")
            torchvision.utils.save_image(qp32_img, args.trainpath + str(ep) + "_qp32_img.png")
            torchvision.utils.save_image(qp42_img, args.trainpath + str(ep) + "_qp42_img.png")
            torchvision.utils.save_image(QP52_img, args.trainpath + str(ep) + "_QP52_img.png")


            loss_gt, loss_32, loss_42 = compute_loss(gt_img, qp32_img, qp42_img, pred_gt, pred_32, pred_42)

            loss_total = loss_gt  + loss_32 + loss_42


            loss_total.backward()
            optimG.step()

            loss_gt_train += [loss_gt.item()]
            loss_42_train += [loss_42.item()]
            loss_32_train += [loss_32.item()]

            loss_total_train += [loss_total.item()]

            print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                  'loss_gt: %.4f loss_42: %.4f loss_32: %.4f '
                  % (ep, idx, num_batch_train, mean(loss_gt_train), mean(loss_42_train),
                     mean(loss_32_train)))

            save(args.dir_chck, G1A_B, G2A_B, G3A_B, optimG, ep)

        # if torch.cuda.current_device() == 0:
        #     print("GPU{} Total_loss:".format(torch.cuda.current_device()), loss)

        scheduler_G.step()



        writer.add_scalars("loss", {"train": mean(loss_total_train)}, ep + 1)
        writer.export_scalars_to_json("./loss.json")
        writer.close()

        if (ep + 1) % args.save_per_epoch == 0:
            # Save model
            save(args.dir_chck, G1A_B, G2A_B, G3A_B, optimG, ep)

    print('Finished training')


if __name__ == "__main__":
    main()
