import sys
import torch.backends.cudnn as cudnn

sys.path.append('../')
# sys.path.append('/root/code/Vessel_Net')
sys.path.append('/root/My/CS_Net_master')
sys.path.append('../data_process/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')

import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from tqdm import tqdm
# from torchinfo import summary
# from thop import profile
# from thop import clever_format



from time import time
from only_for_vessel_seg.data_process.data_load import ImageFolder, get_drive_data
# from only_for_vessel_seg.networks.common.unet_baseline import UNet
# from only_for_vessel_seg.networks.MESnet.backbone import MES_Unet
# from only_for_vessel_seg.networks.common.res2unet import MSUnet
from only_for_vessel_seg.train_test.losses import loss_ce, loss_ce_ds, CF_Loss
from only_for_vessel_seg.train_test.eval_test import val_vessel
from torch.utils.tensorboard import SummaryWriter
from only_for_vessel_seg.train_test.help_functions import platform_info, check_size
from only_for_vessel_seg.train_test.evaluations import threshold_by_otsu
# import only_for_vessel_seg.train_test.losses as loss_library
#
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_1 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_2 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_2
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_3 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_3
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_4 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_4
from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_5 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_6 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_6
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_7 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_7
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_8 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_8
#
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_5_S1 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5_1
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_5_S2 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5_2
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_5_S3 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5_3
# from only_for_vessel_seg.New_Net.Net.New_train_No1.UUU_C_5_S4 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5_4
#
#
# from only_for_vessel_seg.New_Net.Net_1.UU_C_1 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_1
# from only_for_vessel_seg.New_Net.Net_1.UU_C_2 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_2
# from only_for_vessel_seg.New_Net.Net_1.UU_C_3 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_3
# from only_for_vessel_seg.New_Net.Net_1.UU_C_4 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_4
# from only_for_vessel_seg.New_Net.Net_1.UU_C_5 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_5
# from only_for_vessel_seg.New_Net.Net_1.UU_C_6 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_6
from only_for_vessel_seg.New_Net.Net_1.UU_C_7 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7
from only_for_vessel_seg.networks.comparison.res2unet import MSUnet


# from only_for_vessel_seg.networks.common.csnet import CSNet
# from only_for_vessel_seg.networks.common.Unet_Res_BN_CS_C import ResNeSt_CS_BN_C_3
# from only_for_vessel_seg.networks.common.CS_BR import CSNet_BR
# from only_for_vessel_seg.networks.common.Unet_Res_BN_CS_A import ResNeSt_CS_BN_A_2
# from only_for_vessel_seg.networks.common.CS_BR1 import CSNet_BR1
# from only_for_vessel_seg.networks.common.CSNet_resaspp import CSNet_BR_Resp
# from only_for_vessel_seg.networks.common.CS_REAP_AB import CSNet_BR_RP_AB
# from only_for_vessel_seg.networks.common.CS_REAP_AB_test import CSNet_BR_RP_AB_1
# from only_for_vessel_seg.networks.common.CS_REAP_AB_test_bn import CSNet_BR_RP_AB_BN_1
#
# from only_for_vessel_seg.networks.ablation.ablation_unet import UNet
# from only_for_vessel_seg.networks.ablation.ablation_backbone import Backbone_Net
# from only_for_vessel_seg.networks.ablation.ablation_backbone_resaspp import Backbone_resaspp_Net
# from only_for_vessel_seg.networks.ablation.ablation_csam import Backbone_CSAM_Net
# from only_for_vessel_seg.networks.ablation.ablaiton_mta import Backbone_MTA_Net
# from only_for_vessel_seg.networks.ablation.ablation_backbone_A_resaspp import Backbone_A_resaspp_Net
# from only_for_vessel_seg.networks.ablation.ablation_backbone_resaspp_mta import Backbone_resaspp_mta_Net
# from only_for_vessel_seg.networks.ablation.ablation_backbone_A_csam import Backbone_A_CSAM_Net
# from only_for_vessel_seg.networks.comparison.DU_Net import DUNetV1V2
# from only_for_vessel_seg.networks.comparison.CE_Net import CE_Net
# # from only_for_vessel_seg.Net.train_No_1.UUU_C_1 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6
# from only_for_vessel_seg.Net.train_No1.UUU_C_6 import BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6


# from only_for_vessel_seg.FMA_1.abltion.baseline import baseline
# from only_for_vessel_seg.FMA_1.abltion.baseline_bc import baseline_bc
# from only_for_vessel_seg.FMA_1.abltion.baseline_abc import baseline_abc
#
#
# from only_for_vessel_seg.FMA_1.contrast.LadderNetv65 import LadderNetv6
# from only_for_vessel_seg.FMA_1.contrast.DU_Net import DU_Net
# from only_for_vessel_seg.FMA_1.contrast.csnet import CSNet
# from only_for_vessel_seg.FMA_1.contrast.AttnUnet import AttU_Net
#
#
# from only_for_vessel_seg.FMA_1.ablation_4.D import D
# from only_for_vessel_seg.FMA_1.ablation_4.S import S
# from only_for_vessel_seg.FMA_1.ablation_4.D_S_M import D_S_M
#
#
# from only_for_vessel_seg.Net.ablation.without_FMCD import without_FMCD
# from only_for_vessel_seg.Net.ablation.without_Shallow import without_Shallow
#
# from only_for_vessel_seg.networks.ablation.ablation_1cbam import Backbone_1CBAM_Net
# from only_for_vessel_seg.Study.U_Net_res import U_net_res
# from only_for_vessel_seg.Study.CS_Net import CSNet_BR_Resp2

# from only_for_vessel_seg.New_Net.Net.ablation.UUU_C_5_Baseline import UUU_C_5_Baseline
# from only_for_vessel_seg.New_Net.Net.Parameter.UU_C_5_Deep_1 import UU_C_5_Deep_1
# from only_for_vessel_seg.New_Net.Net.Parameter.UU_C_5_Deep_2 import UU_C_5_Deep_2
# from only_for_vessel_seg.New_Net.Net.Parameter.UU_C_5_Deep_3 import UU_C_5_Deep_3



from only_for_vessel_seg import Constants

learning_rates = Constants.learning_rates
gcn_model = False

# adjust learning rate (poly)
def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = base_lr * (1 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_lr1(optimizer, old_lr, ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = old_lr / ratio
    print('update learning rate: %f -> %f' % (old_lr, old_lr / ratio))
    return old_lr / ratio


def update_lr2(epoch, optimizer, total_epoch=Constants.TOTAL_EPOCH):
    new_lr = learning_rates * (1 - epoch / total_epoch)
    for p in optimizer.param_groups:
        p['lr'] = new_lr


# def optimizer_net(net, optimizers, criterion, images, masks,ch):
#     optimizers.zero_grad()
#     # pred, pred1, pred2 = net(images)
#     pred = net(images)
#     loss0 = loss_ce(pred, masks, criterion, ch)
#     # loss1 = loss_ce(pred1, masks, criterion, ch)
#     # loss2 = loss_ce(pred2, masks, criterion, ch)
#     # loss = loss0 + loss1 + loss2
#     loss = loss0
#     loss.backward()
#     optimizers.step()
#     return pred, loss

def optimizer_net(net, optimizers, criterion, images, masks,ch):
    optimizers.zero_grad()
    pred, pred1, pred2 = net(images)
    beta = 1.1
    alpha = 0.03
    gamma = 0.02
    # Initialize CF_Loss with the required parameters
    cf_loss_fn = CF_Loss(beta=beta, alpha=alpha, gamma=gamma)
    # Calculate the loss using CF_Loss
    loss0 = cf_loss_fn(pred, masks)
    loss1 = cf_loss_fn(pred1, masks)
    loss2 = cf_loss_fn(pred2, masks)
    loss = loss0 + loss1 + loss2
    loss.backward()
    optimizers.step()
    return pred, loss



def visual_preds(preds, is_preds=True):  # This for multi-classification
    rand_arr = torch.rand(size=(preds.size()[1], preds.size()[2], 3))
    color_preds = torch.zeros_like(rand_arr)
    outs = preds.permute((1, 2, 0))  # N H W C
    if is_preds is True:
        outs_one_hot = torch.argmax(outs, dim=2)
    else:
        outs_one_hot = outs.reshape((preds.size()[1], preds.size()[2]))
    for H in range(0, preds.size()[1]):
        for W in range(0, preds.size()[2]):
            if outs_one_hot[H, W] == 1:
                color_preds[H, W, 0] = 255
            if outs_one_hot[H, W] == 2:
                color_preds[H, W, 1] = 255
            if outs_one_hot[H, W] == 3:
                color_preds[H, W, 2] = 255
            if outs_one_hot[H, W] == 4:
                color_preds[H, W, 0] = 255
                color_preds[H, W, 1] = 255
                color_preds[H, W, 2] = 255
    return color_preds.permute((2, 0, 1))


def train_model(learning_rates):

    writer = SummaryWriter(comment=f"MyDRIVETrain01", flush_secs=1)
    tic = time()
    loss_lists = []
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    ch = Constants.BINARY_CLASS
    criterion = nn.BCELoss()

    # criterion2 = nn.CrossEntropyLoss()
    #net = ResNeSt_CS_BN_C_3(1, ch).to(device)
    net = BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU_7(1, ch).to(device)  # 1
    # net = without_Shallow(1, ch).to(device) # 1

    # net = CE_Net(1, ch).to(device)  # 1
    # net = DUNetV1V2(1, ch).to(device)  # 1
    # net = UNet(1, ch).to(device)  # 1
    # net = CSNet_BR_RP_AB(1, ch).to(device)  # 1
    # summary(net, input_data=torch.rand(Constants.BATCH_SIZE, 1, 512, 512))

    # inputs = torch.randn(1, 1, 224, 224)  ####(360,640)
    # inputs = inputs.to(device)
    # flops, params = profile(net, inputs=(inputs,))  ##verbose=False
    # # flops, params = clever_format([flops, params], '%3.f')
    # # print('The number of MACs is %s' % (flops / 1e9))  ##### MB
    # # print('The number of params is %s' % (params / 1e6))  ##### MB
    # # print(flops, params)
    # # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # # print("params=", str(params / 1e6) + '{}'.format("M"))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))



    # if device == 'cuda':
    #     net.cuda()
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True

    optimizers = optims.Adam(net.parameters(), lr=learning_rates, betas=(0.9, 0.999))
    trains, val = get_drive_data()
    dataset = ImageFolder(trains[0], trains[1])
    data_loader = data.DataLoader(dataset, batch_size= Constants.BATCH_SIZE, shuffle=True, num_workers=0)
    rand_img, rand_label, rand_pred = None, None, None
    for epoch in range(1, total_epoch + 1):
        net.train(mode=True)
        data_loader_iter = iter((data_loader))
        train_epoch_loss = 0
        index = 0
        for img, mask in data_loader_iter:
            # check_size(img, mask, mask)
            img = img.to(device)
            mask = mask.to(device)
            pred, train_loss = optimizer_net(net, optimizers, criterion, img, mask, ch)
            train_epoch_loss += train_loss.item()
            index = index + 1
            if np.random.rand(1) > 0.4 and np.random.rand(1) < 0.8:
                rand_img, rand_label, rand_pred = img, mask, pred

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        writer.add_scalar('Train/loss', train_epoch_loss, epoch)
        if ch ==1:      # for [N,1,H,W]
            rand_pred_cpu = rand_pred[0, :, :, :].detach().cpu().reshape((-1,)).numpy()
            # threshold = 0.5
            # rand_pred_cpu[rand_pred_cpu >= threshold] = 1
            # rand_pred_cpu[rand_pred_cpu <  threshold] = 0
            rand_pred_cpu = threshold_by_otsu(rand_pred_cpu)
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,)).numpy()
            writer.add_scalar('Train/acc', rand_pred_cpu[np.where(new_mask == rand_pred_cpu)].shape[0] / new_mask.shape[0], epoch)  # for [N,H,W,1]
        if ch ==2:      # for [N,2,H,W]
            new_mask = rand_label[0, :, :, :].cpu().reshape((-1,))
            new_pred = torch.argmax(rand_pred[0, :, :, :].permute((1, 2, 0)), dim=2).detach().cpu().reshape((-1,))
            t = new_pred[torch.where(new_mask == new_pred)].size()[0]
            writer.add_scalar('Train/acc', t / new_pred.size()[0], epoch)

        platform_info(epoch, tic, train_epoch_loss, Constants.IMG_SIZE, optimizers)
        if epoch % 10 == 1:
            writer.add_image('Train/image_origins', rand_img[0, :, :, :], epoch)
            writer.add_image('Train/image_labels', rand_label[0, :, :, :], epoch)
            if ch == 1:  # for [N,1,H,W]
                writer.add_image('Train/image_predictions', rand_pred[0, :, :, :], epoch)
            if ch == 2:  # for [N,2,H,W]
                  writer.add_image('Train/image_predictions', torch.unsqueeze(torch.argmax(rand_pred[0, :, :, :], dim=0), 0),
                             epoch)
        update_lr2(epoch, optimizers)  # modify  lr
        # adjust_lr(optimizers, base_lr=0.001, iter=epoch, max_iter=600, power=0.9)

        print('************ start to validate current model {}.iter performance ! ************'.format(epoch))
        acc, sen, f1score, val_loss = val_vessel(net, val[0], val[1], val[0].shape[0], epoch)
        writer.add_scalar('Val/accuracy', acc, epoch)
        writer.add_scalar('Val/sensitivity', sen, epoch)
        writer.add_scalar('Val/f1score', f1score, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)

        model_name = Constants.saved_path + "{}.iter3".format(epoch)
        torch.save(net, model_name)

        # if train_epoch_loss >= train_epoch_best_loss:
        #     no_optim += 1
        # else:
        #     no_optim = 0
        #     train_epoch_best_loss = train_epoch_loss
        #     model_name = Constants.saved_path + "{}.iter2".format(epoch)
        #     torch.save(net, model_name)
        # if no_optim > Constants.NUM_EARLY_STOP:
        #     print('Early stop at %d epoch' % epoch)
        #     break
        # if epoch % 20 == 0 and epoch != 0:
        #     model_name = Constants.saved_path + "{}.iter2".format(epoch)
        #     torch.save(net, model_name)
    print('***************** Finish training process ***************** ')

if __name__ == '__main__':
    train_model(learning_rates)
    pass

