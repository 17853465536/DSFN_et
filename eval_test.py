import sys
import torch.backends.cudnn as cudnn
# sys.path.append('/root/code/Vessel_Net')
sys.path.append('/root/My/CS_Net_master')
sys.path.append('../data_process/')
sys.path.append('../networks/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')
sys.path.append('../networks/threestage/')

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from torch.autograd import Variable as V
import sklearn.metrics as metrics
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from only_for_vessel_seg.data_process.data_ultils import read_all_images
from only_for_vessel_seg.data_process.data_load import ImageFolder,get_drive_data,get_drive_data1
from only_for_vessel_seg.networks.common.unet_baseline import UNet
# from avseg import AVSegNet
from only_for_vessel_seg import Constants
from only_for_vessel_seg.train_test.evaluations import misc_measures,roc_pr_curve,threshold_by_otsu
from only_for_vessel_seg.test.differentce_retinal import retina_color_different

import warnings
warnings.filterwarnings('ignore')
BATCHSIZE_PER_CARD = 1


def load_model(path):
    print(' [*] Reading checkpoint...')
    net = torch.load(path)
    return net

def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))          # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def val_vessel(net, imgs, masks, length, epoch =0, ch = Constants.BINARY_CLASS):
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis, val_loss = [],[],[],[],[],[],[],[],[],[]
    net.eval()
    with torch.no_grad():
        for iteration in range(0, length):
            x_img = imgs[iteration]
            x_img = np.expand_dims(x_img, axis=0)                     # (H, W, C) to (1, H, W, C)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)


            print(x_img.size(),'---------------')
            # generated_vessel = net(x_img)
            # generated_vessel = crop_eval(net,x_img)
            generated_vessel = crop_eval_new_V1(net, x_img)
            vl = nn.BCELoss()(generated_vessel.detach().cpu().reshape((-1,)), torch.tensor(masks[iteration].reshape((-1,)), dtype=torch.float))
            val_loss.append(vl.numpy())
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
            if ch ==1:   # for [N,1,H,W]
                visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + 'val_prob_pic' + str(iteration))
                threshold = 0.5
                generated_vessel[generated_vessel >= threshold] = 1
                generated_vessel[generated_vessel <  threshold] = 0
            if ch ==2:   # for [N,H,W,2]
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis =3), axis=3)
            generated_vessel = np.squeeze(generated_vessel, axis=0)   # (1, H, W, 1) to (H, W, 1)
            visualize(np.asarray(generated_vessel), Constants.visual_results + 'val_pic' + str(iteration))
            # print('value check :', np.max(masks[iteration]), str(iteration), np.min(masks[iteration]))
            metrics_current = misc_measures(masks[iteration].reshape((-1,)), generated_vessel.reshape((-1,)), False)
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])
        print('********************** below is validation evaluation of epoch {} results **********************'.format(epoch))
        print('Accuracy average is:{}, std is:{}'.format(np.mean(acc), np.std(acc)))
        print('Sensitivity average is:{}, std is:{}'.format(np.mean(sensitivity), np.std(sensitivity)))
        print('Specificity average is:{}, std is:{}'.format(np.mean(specificity), np.std(specificity)))
        print('Precision average is:{}, std is:{}'.format(np.mean(precision), np.std(precision)))
        print('G average is:{}, std is:{}'.format(np.mean(G), np.std(G)))
        print('F1_score average is:{}, std is:{}'.format(np.mean(F1_score), np.std(F1_score)))
        print('Mse average is:{}, std is:{}'.format(np.mean(mse), np.std(mse)))
        print('Iou average is:{}, std is:{}'.format(np.mean(iou), np.std(iou)))
        print('Hausdorff_distance average is:{}, std is:{}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis)))
        s = 'epoch:{}, Accuracy average is:{}, Sensitivity average is:{}, F1_score average is:{}, Iou average is:{}'.format(
            epoch,np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(iou))
        with open(os.path.join('/root/My/CS_Net_master/only_for_vessel_seg/log', 'save_result.txt'), 'a',
                  encoding='utf-8') as f:
            f.write(s)
            f.write('\n')

        # with open(os.path.join('/root/code/Vessel_Net/only_for_vessel_seg/log', 'save_result.txt'), 'a',
        #           encoding='utf-8') as f:
        #     f.write(s)
        #     f.write('\n')

    return np.mean(acc), np.mean(sensitivity), np.mean(F1_score), np.mean(val_loss)


def test_vessel(path, ch = Constants.BINARY_CLASS):
    images, masks = get_drive_data(is_train=False)
    acc, sensitivity, specificity, precision, G, F1_score, mse, iou, hausdorff_dis = [], [], [], [], [], [], [], [], []
    pr_g, pr_l = [], []

    with torch.no_grad():
        net = load_model(path)
        net.eval()
        for iter_ in range(int(Constants.Classes_drive_color)):
            import torch.nn as nn
            x_img = images[iter_]
            x_img = np.expand_dims(x_img, axis=0)
            x_img = torch.tensor(x_img, dtype=torch.float32).to(device)
            # generated_vessel = net(x_img)
            # generated_vessel = crop_eval(net,x_img)
            generated_vessel = crop_eval_new_V1(net, x_img)
            generated_vessel = generated_vessel.permute((0, 2, 3, 1)).detach().cpu().numpy()
    
            if ch == 1:  # for [N,1,H,W]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
                visualize(np.asarray(generated_vessel[0, :, :, :, ]), Constants.visual_results + str(iter_) + 'prob')
                threshold = 0.5                                                               # for [N,H,W,1]
                generated_vessel[generated_vessel >= threshold] = 1
                generated_vessel[generated_vessel < threshold] = 0
#                 generated_vessel = threshold_by_otsu(generated_vessel)
            if ch == 2:  # for [N,H,W,2]
                generated_vessel = np.expand_dims(np.argmax(generated_vessel, axis=3), axis=3)  # for [N,H,W,2]
                pr_g.append(generated_vessel.reshape((-1,)).tolist())
                pr_l.append(masks[iter_].reshape((-1,)).tolist())
            generated_vessel = np.squeeze(generated_vessel, axis=0)  # (1, H, W, 1) to (H, W)
            visualize(np.asarray(generated_vessel),Constants.visual_results + str(iter_)+ 'seg')
            retina_color_different(np.asarray(generated_vessel), masks[iter_].transpose((1, 2, 0)),
                                   Constants.visual_results + str(iter_) + 'different') #different map
            print('value check :', np.max(masks[iter_]), str(iter_), np.min(masks[iter_]))
            metrics_current = misc_measures(masks[iter_].reshape((-1,)), generated_vessel.reshape((-1,)))
    
            acc.append(metrics_current[0])
            sensitivity.append(metrics_current[1])
            specificity.append(metrics_current[2])
            precision.append(metrics_current[3])
            G.append(metrics_current[4])
            F1_score.append(metrics_current[5])
            mse.append(metrics_current[6])
            iou.append(metrics_current[7])
            hausdorff_dis.append(metrics_current[8])
            print('image: {} test evaluations **** acc is: {}, sensitivity is: {},specificity is: {},precision is: {},G is: {},F1_score is: {},'
                  'mse is: {},iou is: {},hausdorff is: {} ****'.format(iter_, metrics_current[0],metrics_current[1],metrics_current[2],metrics_current[3],
                                                             metrics_current[4],metrics_current[5],metrics_current[6],metrics_current[7],
                                                             metrics_current[8]))
        AUC_prec_rec, AUC_ROC = roc_pr_curve(np.array(pr_l).reshape((-1,)), np.array(pr_g).reshape((-1,)),
                                             Constants.visual_results)
        path_files_saved = Constants.visual_results + 'evaluation.txt'
        print('********************** final test results has been saved in to {} **********************'.format(path_files_saved))
        str_a = 'Area of PR curve is: {}, Area of ROC curve is: {}'.format(AUC_prec_rec, AUC_ROC)
        str_b = 'Accuracy average is: {}, std is: {}'.format(np.mean(acc), np.std(acc))
        str_c = 'Sensitivity average is: {}, std is: {}'.format(np.mean(sensitivity), np.std(sensitivity))
        str_d = 'Specificity average is: {}, std is: {}'.format(np.mean(specificity), np.std(specificity))
        str_e = 'Precision average is: {}, std is: {}'.format(np.mean(precision), np.std(precision))
        str_f = 'G average is: {}, std is: {}'.format(np.mean(G), np.std(G))
        str_g = 'F1_score average is:{}, std is: {}'.format(np.mean(F1_score), np.std(F1_score))
        str_h = 'Mse average is: {}, std is: {}'.format(np.mean(mse), np.std(mse))
        str_i = 'Iou average is: {}, std is: {}'.format(np.mean(iou), np.std(iou))
        str_j = 'Hausdorff_distance average is: {}, std is: {}'.format(np.mean(hausdorff_dis), np.std(hausdorff_dis))

        f = open(path_files_saved, 'w', encoding='utf-8')
        f.write(str_a+'\n')
        f.write(str_b+'\n')
        f.write(str_c+'\n')
        f.write(str_d+'\n')
        f.write(str_e+'\n')
        f.write(str_f+'\n')
        f.write(str_g+'\n')
        f.write(str_h+'\n')
        f.write(str_i+'\n')
        f.write(str_j+'\n')
        f.close()

def crop_eval(net, image, crop_size = Constants.resize_drive):
    '''
    :param net:
    :param image:     image is tensor form of [N, C, H, W], size is (584, 565)
    :param crop_size: 512 default
    :return:          584 , 565
    '''
    # print('val image size is:'.format(image.size()))
    d_h, d_w, h, w =  image.size()[2] - crop_size, image.size()[3] - crop_size, image.size()[2],image.size()[3]
    crop_lu_im = image[:, :, 0:h - d_h, 0:w - d_w]
    crop_ld_im = image[:, :, 0:h - d_h, d_w:w]
    crop_ru_im = image[:, :, d_h:h, 0:w - d_w]
    crop_rd_im = image[:, :, d_h:h, d_w:w]
    # lu, ru, ld, rd = net(crop_lu_im), net(crop_ru_im), net(crop_ld_im), net(crop_rd_im)
    lu, lu1, lu2 = net(crop_lu_im)
    ru, ru1, ru2 = net(crop_ru_im)
    ld, ld1, ld2 = net(crop_ld_im)
    rd, rd1, rd2 = net(crop_rd_im)
    new_image = torch.zeros_like(torch.unsqueeze(image[:, 0, :, :, ], dim=1))
    # new_image = torch.zeros_like(image)

    for i in range(0, h):
        for j in range(0, w):
            if i>=d_h and j >=d_w and i<crop_size and j<crop_size:
                new_image[:,:,i,j] = (lu[:,:,i,j] + ld[:,:,i,j-d_w] + ru[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w]) /4

            if i>=0 and j >=0 and i<d_h and j<d_w:
                new_image[:, :, i, j] = lu[:,:,i,j]
            if i>=0 and j >=d_w and i<d_h and j<crop_size:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ld[:,:,i,j-d_w])/2
            if i>=0 and j >=crop_size and i<d_h:
                new_image[:, :, i, j] = ld[:,:,i,j-d_w]
            if i>=d_h and j >=0 and i<crop_size and j<d_w:
                new_image[:, :, i, j] = (lu[:,:,i,j] + ru[:,:,i-d_h,j])/2
            if i>=d_h and j >=crop_size and i<crop_size:
                new_image[:, :, i, j] = (ld[:,:,i,j-d_w] + rd[:,:,i-d_h,j-d_w])/2

            if i>=crop_size and j >=0 and j<d_w:
                new_image[:, :, i, j] = ru[:,:,i-d_h,j]
            if i>=crop_size and j>=d_w and j <crop_size :
                new_image[:, :, i, j] = (ru[:,:,i-d_h,j] + rd[:,:,i-d_h,j-d_w])/2
            if i>=crop_size and j >crop_size:
                new_image[:, :, i, j] = rd[:,:,i-d_h,j-d_w]

    return new_image.to(device)


# def crop_eval_new_V2(net, image, crop_size=512, row=4, col=4):
#     h_o, w_o = image.size()[2], image.size()[3]
#     assert h_o % crop_size == 0 and w_o % crop_size == 0, "Image dimensions must be divisible by crop size."
#
#     # 初始化分块结果
#     group_image = torch.zeros((row, col, image.size(0), image.size(1), crop_size, crop_size)).to(device)
#     if net is not None:
#         group_image1 = torch.zeros_like(group_image).to(device)
#         group_image2 = torch.zeros_like(group_image).to(device)
#
#     step_h = crop_size
#     step_w = crop_size
#
#     merge_img = torch.zeros_like(image).to(device)
#
#     # 切割图像
#     for m in range(row):
#         for n in range(col):
#             dh = m * step_h
#             dw = n * step_w
#             if net is None:
#                 group_image[m, n] = image[:, :, dh:dh + crop_size, dw:dw + crop_size]
#             else:
#                 group_image[m, n], group_image1[m, n], group_image2[m, n] = net(
#                     image[:, :, dh:dh + crop_size, dw:dw + crop_size])
#
#     # 合并切块预测结果
#     for i in range(h_o):
#         for j in range(w_o):
#             p = i // crop_size
#             q = j // crop_size
#             p_h = i % crop_size
#             p_w = j % crop_size
#             merge_img[:, :, i, j] = group_image[p, q, :, :, p_h, p_w]
#
#     print('================= has finished this picture prediction ==================')
#     return merge_img[:, :, 0:h_o, 0:w_o].to(device)


def crop_eval_new_V1(net, image, crop_size = 960, row=3, col=4):
    h_o, w_o = image.size()[2], image.size()[3]
    group_image = torch.zeros_like(torch.rand(size=(row, col, image.size()[0], image.size()[1], crop_size,  crop_size))).to(device)
    group_image1 = torch.zeros_like(
        torch.rand(size=(row, col, image.size()[0], image.size()[1], crop_size, crop_size))).to(device)
    group_image2 = torch.zeros_like(
        torch.rand(size=(row, col, image.size()[0], image.size()[1], crop_size, crop_size))).to(device)
    image, s_h, s_w= padding_img(image, crop_size, row, col)
    h, w = image.size()[2], image.size()[3]
    assert (crop_size > s_h and crop_size > s_w)
    merge_img = torch.zeros_like(image).to(device)
    for m in range(0, row):
        for n in range(0, col):
            dh = m * s_h
            dw = n * s_w
            if net is None:
                group_image[m,n,:,:,:,:,] = image[:,:,dh:dh+crop_size, dw:dw+crop_size]
            else:
                group_image[m,n,:,:,:,:,], group_image1[m,n,:,:,:,:,], group_image2[m,n,:,:,:,:,]= net(image[:,:,dh:dh+crop_size, dw:dw+crop_size])

    # print(image.size(),'-------------------- here has cropped -----------------------',s_h, s_w, h, w, crop_size)

    for i in range(0, h):
        for j in range(0, w):
           p, q = np.maximum((i - crop_size)// s_h, -1) + 1, np.maximum((j - crop_size)// s_w, -1) + 1
           p_h, p_w = i - p * s_h, j - q * s_w
           # print(p, q, p_h, p_w)
           j_s = 1
           merge_img[:, :, i, j] = group_image[p, q, :, :, p_h, p_w]
           for k1 in range(0, row):
               for k2 in range(0, col):
                   if (k1 < p_h / s_h and k1 < row - p):
                       merge_img[:,:, i, j] += group_image[p + k1, q, :, :, p_h - k1 * s_h, p_w]
                       j_s +=1
                   if (k1 < (crop_size-p_h) / s_h and k1 <= p):
                       merge_img[:, :, i, j] += group_image[p - k1, q, :, :, p_h + k1 * s_h, p_w]
                       j_s += 1
                   if (k2 < p_w / s_w and k2 < col - q):
                       merge_img[:,:, i, j] += group_image[p, q + k2, :, :, p_h, p_w - k2 * s_w]
                       j_s += 1
                   if (k2 < (crop_size - p_w) / s_w and k2 <= q):
                       merge_img[:,:, i, j] += group_image[p, q - k2, :, :, p_h, p_w + k2 * s_w]
                       j_s += 1
           merge_img[:, :, i, j] /= j_s
    print('================= has finished this picture prediction ================== ')
    return merge_img[:, :, 0:h_o, 0:w_o].to(device)

def padding_img(image, crop_size, rows, clos):
    pad_h, pad_w = (image.size()[2] - crop_size)%(rows-1), (image.size()[3] - crop_size)%(clos-1)
    image = padding_hw(image, dims='h', ns= 0 if pad_h==0 else rows -1 -pad_h)
    image = padding_hw(image, dims='w', ns= 0 if pad_w==0 else clos -1 -pad_w)
    return image.to(device), (image.size()[2] - crop_size)//(rows-1), (image.size()[3] - crop_size)//(clos-1)

def padding_hw(img, dims = 'h', ns = 0):
    if ns ==0:
        return img
    else:
        after_expanding = None
        if dims == 'h':
            pad_img = torch.zeros_like(img[:,:,0 : ns,:,])
            after_expanding = torch.cat([img, pad_img], dim=2)
        elif dims == 'w':
            pad_img = torch.zeros_like(img[:,:,:,0:ns])
            after_expanding = torch.cat([img, pad_img], dim=3)
        return after_expanding



# def writer_into_files(path = Constants.visual_results+'evaluation.txt'):
#     f = open(path, 'w', encoding='utf-8')
#     f.write('ewe{}'.format(12)
#     f.close()

if __name__ == '__main__':
    path = '../log/weights_save/HRF_(UU_C_1)_3_(960_768_512)/12.iter3'
    test_vessel(path)

    pass