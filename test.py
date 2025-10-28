import cv2
import torch
import glob
import os
import time
import thop
from thop import profile
import torchvision
from args import args
from Network import generator, ResNet, UnetDiscriminotor, init_net
import numpy as np
from PIL import Image, ImageDraw

# parameters

def rgb_to_yuv420(rgb):
    rgb_array = np.array(rgb)
    H, W, _ = rgb_array.shape
    yuv_array = np.empty((H, W, 3), dtype = np.uint8)

    yuv_array[:, :, 0] = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
    yuv_array[:, :, 1] = -0.14713 * rgb_array[:, :, 0] - 0.288862 * rgb_array[:, :, 1] + 0.436 * rgb_array[:, :, 2]
    yuv_array[:, :, 2] = 0.615 * rgb_array[:, :, 0] - 0.51498 * rgb_array[:, :, 1] - 0.10001 * rgb_array[:, :, 2]


    return Image.fromarray(yuv_array)

def yuv420_to_rgb(yuv):
    yuv_array = np.array(yuv)
    H, W, _ = yuv_array.shape
    rgb_array = np.empty((H, W, 3), dtype=np.uint8)

    rgb_array[:, :, 0] = yuv_array[:, :, 0] + 1.13983 * yuv_array[:, :, 2]
    rgb_array[:, :, 1] = yuv_array[:, :, 0] -0.39465 * yuv_array[:, :, 1] -0.5806 * yuv_array[:, :, 2]
    rgb_array[:, :, 2] = yuv_array[:, :, 0] + 2.03211 * yuv_array[:, :, 1]




def loadG(dir_chck, G1A_B, G2A_B, G3A_B, optimG=[], epoch=[],
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


        return G1A_B, G2A_B, G3A_B, optimG, epoch

    elif mode == 'test':

        G1A_B.load_state_dict(dict_net['G1A_B'])
        G2A_B.load_state_dict(dict_net['G2A_B'])
        G3A_B.load_state_dict(dict_net['G3A_B'])

        return  G1A_B, G2A_B, G3A_B, epoch

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

G1A_B = generator(2, 1, 64, 6).cuda(args.gpu)
G2A_B = generator(2, 1, 64, 6).cuda(args.gpu)
G3A_B = generator(2, 1, 64, 6).cuda(args.gpu)
# GB_A = generator(1, 1, 64, 6).cuda(args.gpu)

# 创建一个测试输入（假设输入图像大小为 128x128）



is_cuda = torch.cuda.is_available()

G1A_B.eval()
G2A_B.eval()
G3A_B.eval()
# GB_A.eval()


G1A_B,G2A_B,G3A_B,st_epoch = loadG(args.dir_chck, G1A_B, G2A_B, G3A_B, epoch=23, mode="test")
# GB_A, st_epoch =  loadD(args.dir_chck, GB_A, epoch=15, mode="test")
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model G1A_B Parameters: {count_model_parameters(G1A_B):,}")
print(f"Model G2A_B Parameters: {count_model_parameters(G2A_B):,}")
print(f"Model G3A_B Parameters: {count_model_parameters(G3A_B):,}")
if is_cuda:
    G1A_B.cuda(args.gpu)
    G2A_B.cuda(args.gpu)
    G3A_B.cuda(args.gpu)
    # GB_A.cuda(args.gpu)

qps=["22","32","42","52"]
# qps = ["22"]
for qp in qps:
    for i in range(15,16):
# dataloader

        gt_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/gfvc_v1-AhG16_GFVC_Software_v4/source/experiment/Rec_frames/VOXCELEB_"+f"{i:03d}"+"_"+f"{qp}"+"/"]
        LR_list = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/gfvc_v1-AhG16_GFVC_Software_v4/source/experiment/Rec_frames/VOXCELEB_"+f"{i:03d}"+"_52"+"/"]


        save_path_A  = ["/media/media/6dbfb7d7-4857-4183-acea-f582f662b061/liulu/trainVVC/networkcomplexity/VOXCELEB_"+f"{i:03d}"+"_"+f"{qp}"+"/"]
        if not os.path.exists(save_path_A[0]):
            os.makedirs(save_path_A[0],exist_ok=True)
        os.makedirs(save_path_A[0]+"input/", exist_ok=True)
        os.makedirs(save_path_A[0]+"gt/", exist_ok=True)
        # os.makedirs(save_path_A[0]+"pred_32/", exist_ok=True)
        # os.makedirs(save_path_A[0]+"pred_42/", exist_ok=True)
        os.makedirs(save_path_A[0]+"pred_gt/", exist_ok=True)
        for i in range(0,len(LR_list)):
            LR_inputdir = LR_list[i]
            LR_input = os.listdir(LR_inputdir)
            LR_input.sort()

            gtdir = gt_list[i]
            gt = os.listdir(gtdir)
            gt.sort()
            total_inference_time = 0
            num_samples = len(LR_input)


            for j in range(0, len(LR_input)):
                j = j
                H = 128
                W = 128
                ref_img_3_path = gtdir + gt[1]
                gt_img_3_path = gtdir + gt[j]
                QP52_img_3_path = LR_inputdir + LR_input[j]

                if QP52_img_3_path.endswith(".png") and gt_img_3_path.endswith(".png"):

                    ref_img_3 = Image.open(ref_img_3_path).convert('RGB')
                    QP52_img_3 = Image.open(QP52_img_3_path).convert('RGB')
                    gt_img_3 = Image.open(gt_img_3_path).convert('RGB')
                    gt_img_3 = gt_img_3.resize((H, W), Image.BILINEAR)
                    ref_img_3 = ref_img_3.resize((H, W), Image.BILINEAR)
                    QP52_img_3 = QP52_img_3.resize((H, W), Image.BILINEAR)




                    lr_16_3 = np.array(QP52_img_3)
                    lr_16_3 = cv2.cvtColor(lr_16_3, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path_A[0]+ 'input/' + '{}_QP22_img_3.png'.format(str(j).zfill(4)), lr_16_3)

                    ref_img_3 = np.array(ref_img_3)
                    ref_img_3 = cv2.cvtColor(ref_img_3, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path_A[0]+ 'input/' + '{}_ref_img_3.png'.format(str(j).zfill(4)), ref_img_3)

                    gt_img_3 = np.array(gt_img_3)
                    gt_img_3 = cv2.cvtColor(gt_img_3, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path_A[0]+ 'gt/' + '{}_gt_img_3.png'.format(str(j).zfill(4)), gt_img_3)

                    ref_img_3 = cv2.cvtColor(ref_img_3, cv2.COLOR_BGR2YUV)
                    gt_img_3 = cv2.cvtColor(gt_img_3, cv2.COLOR_BGR2YUV)
                    lr_16_3 = cv2.cvtColor(lr_16_3, cv2.COLOR_BGR2YUV)

                    ref_img_3 = torch.from_numpy(ref_img_3.transpose((2, 0, 1)))
                    ref_img_3 = ref_img_3.float().div(255)

                    gt_img_3 = torch.from_numpy(gt_img_3.transpose((2, 0, 1)))
                    gt_img_3 = gt_img_3.float().div(255)

                    lr_16_3 = torch.from_numpy(lr_16_3.transpose((2, 0, 1)))
                    lr_16_3 = lr_16_3.float().div(255)

                    if is_cuda and args.gpu is not None:
                        ref_img_y = ref_img_3[0, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        ref_img_u = ref_img_3[1, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        ref_img_v = ref_img_3[2, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)

                        gt_img_y = gt_img_3[0, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        gt_img_u = gt_img_3[1, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        gt_img_v = gt_img_3[2, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)

                        lr_16_3_y = lr_16_3[0, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        lr_16_3_u = lr_16_3[1, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)
                        lr_16_3_v = lr_16_3[2, :, :].unsqueeze(0).unsqueeze(0).cuda(args.gpu)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    pred_42_y = G1A_B(torch.cat((lr_16_3_y, ref_img_y), dim=1))
                    pred_32_y = G2A_B(torch.cat((pred_42_y, ref_img_y), dim=1))
                    pred_gt_y = G3A_B(torch.cat((pred_32_y, ref_img_y), dim=1))

                    torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    total_inference_time += inference_time

                    pred_gt = torch.cat((pred_gt_y, gt_img_u, gt_img_v), dim=1)
                    pred_gt = pred_gt.squeeze(0)
                    pred_gt = pred_gt.permute(1,2,0)
                    pred_gt = (pred_gt.cpu()).detach().numpy()
                    pred_gt = cv2.cvtColor(pred_gt, cv2.COLOR_YUV2BGR)
                    cv2.imwrite(save_path_A[0]+ 'pred_gt/'+ '{}_pred_gt.png'.format(str(j).zfill(4)),pred_gt * 255)

                    pred_16 = torch.cat((pred_42_y, gt_img_u, gt_img_v), dim=1)
                    pred_16 = pred_16.squeeze(0)
                    pred_16 = pred_16.permute(1, 2, 0)
                    pred_16 = (pred_16.cpu()).detach().numpy()
                    pred_16 = cv2.cvtColor(pred_16, cv2.COLOR_YUV2BGR)
                    cv2.imwrite(save_path_A[0] + 'pred_42/'+ '{}_pred_42.png'.format(str(j).zfill(4)), pred_16 * 255)

                    pred_8 = torch.cat((pred_32_y, gt_img_u, gt_img_v), dim=1)
                    pred_8 = pred_8.squeeze(0)
                    pred_8 = pred_8.permute(1, 2, 0)
                    pred_8 = (pred_8.cpu()).detach().numpy()
                    pred_8 = cv2.cvtColor(pred_8, cv2.COLOR_YUV2BGR)
                    cv2.imwrite(save_path_A[0] +'pred_32/'+ '{}_pred_32.png'.format(str(j).zfill(4)), pred_8 * 255)
                    print(f"Processed frame {j}/{num_samples}, inference time: {inference_time:.4f} sec")
                    print(j)

                avg_inference_time = total_inference_time / num_samples
                print(f"Average inference time per frame: {avg_inference_time * 1000:.2f} ms")
                # 计算模型的 MACs
    input_tensor = torch.randn(1, 3, 128, 128).cuda()

    macs_G1A_B, params_G1A_B = profile(G1A_B, inputs=(input_tensor,))
    macs_G2A_B, params_G2A_B = profile(G2A_B, inputs=(input_tensor,))
    macs_G3A_B, params_G3A_B = profile(G3A_B, inputs=(input_tensor,))

    # 转换为 KMAC
    kmac_G1A_B = macs_G1A_B / (1e3*128*128)
    kmac_G2A_B = macs_G2A_B /(1e3*128*128)
    kmac_G3A_B = macs_G3A_B / (1e3*128*128)

    print(f"G1A_B: {kmac_G1A_B:.2f} KMACs")
    print(f"G2A_B: {kmac_G2A_B:.2f} KMACs")
    print(f"G3A_B: {kmac_G3A_B:.2f} KMACs")




