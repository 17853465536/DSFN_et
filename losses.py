import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
'''
files to devise different loss strategy !
'''

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()      # multiply flags and labels
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.5, gamma = 2, logits = False, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class weighted_entropy(nn.Module):
    '''
    pred  : N, C,
    label : N, -1
    '''
    def __init__(self, need_soft_max = True):
        super(weighted_entropy, self).__init__()
        self.need_soft_max = need_soft_max
        pass

    def forward(self, pred, label):
        if self.need_soft_max is True:
            preds = F.softmax(pred, dim=1)
        else:
            preds = pred
        epusi  = 1e-10
        counts = torch.rand(size=(2,))
        counts[0] = label[torch.where(label == 0)].size(0)
        counts[1] = label[torch.where(label == 1)].size(0)
        N = label.size()[0]
        weights = counts[1]
        weights_avg = 1 - weights / N
        loss = weights_avg * torch.log(preds[:,1] + epusi) + (1 - weights_avg) * torch.log(1 - preds[:,1] + epusi)
        loss = - torch.mean(loss)
        return loss

def loss_ce(preds, masks, criterion, selected_mode = 1):

    '''
    :param preds:       N  H  W  C
    :param masks:       N  1  H  W
    :param criterion:   This is used to calculate nn.cross-entropy() or nn.BCE-loss(), both is OK !
    :return:            criterion (N*H*W, C  and  N,-1)
    '''
    if selected_mode == 1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs = preds.permute((0, 2, 3, 1))         # N H W C
    outs = outs.reshape((-1, outs.size()[3]))  # N*H*W, C
    if selected_mode == 1:
        outs = outs.reshape((-1,))
    masks = masks.reshape((-1,))               # N,1,H,W ===> N,-1
    if selected_mode == 2:
        masks = torch.tensor(masks, dtype=torch.long)
    return criterion(outs, masks)



# def encode_mask(ground_truth):
#     # 将ground_truth转换为二分类（0:背景, 1:血管）
#     encode_tensor = ground_truth.to(torch.float32)  # 转为float32类型
#     encode_tensor = encode_tensor.to(device=torch.device('cuda'))  # 将数据移至GPU
#     return encode_tensor
#
#
# class CF_Loss(nn.Module):
#     def __init__(self, beta=1.1, alpha=0.05, gamma=0.08, selected_mode=1):
#         super(CF_Loss, self).__init__()
#         self.beta = beta  # Binary Cross-Entropy Loss的权重
#         self.alpha = alpha  # 特征分布损失的权重
#         self.gamma = gamma  # 血管密度损失的权重
#         img_size = 512
#         self.p = torch.tensor(img_size, dtype=torch.float)
#         self.n = torch.log(self.p) / torch.log(torch.tensor([2]).to('cuda'))
#         self.n = torch.floor(self.n)
#         self.sizes = 2 ** torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)
#
#         # 选取损失函数类型：1 为 BCELoss，2 为 CrossEntropyLoss
#         self.selected_mode = selected_mode
#         if self.selected_mode == 1:
#             self.criterion = nn.BCELoss()
#         elif self.selected_mode == 2:
#             self.criterion = nn.CrossEntropyLoss()
#
#
#     def get_count(self, sizes, p, masks_pred_sigmoid):
#         counts = torch.zeros((masks_pred_sigmoid.shape[0], len(sizes), 1))
#         index = 0
#
#         for size in sizes:
#             stride = (size, size)
#             pad_size = torch.where((p % size) == 0, torch.tensor(0, dtype=torch.int),
#                                    (size - p % size).to(dtype=torch.int))
#             pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
#             pool = nn.AvgPool2d(kernel_size=(size, size), stride=stride)
#
#             S = pad(masks_pred_sigmoid)
#             S = pool(S)
#             S = S * ((S > 0) & (S < (size * size)))
#
#             counts[..., 0, 0] = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / (S[:, 1, ...] > 0).sum()
#
#         return counts
#
#     def loss_ce(self, preds, masks):
#         '''
#         :param preds:       N  H  W  C (模型预测)
#         :param masks:       N  1  H  W (真实标签)
#         :return:            返回根据选定模式计算的损失值
#         '''
#         if self.selected_mode == 1:  # 使用 BCELoss 时，masks 应为 float 类型
#             masks = torch.tensor(masks, dtype=torch.float)
#
#         outs = preds.permute((0, 2, 3, 1))  # 调整预测输出形状为 N H W C
#         outs = outs.reshape((-1, outs.size()[3]))  # 调整为 (N*H*W, C)
#
#         if self.selected_mode == 1:
#             outs = outs.reshape((-1,))  # BCELoss 需要展平成单通道
#         masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1
#
#         if self.selected_mode == 2:  # 使用 CrossEntropyLoss 时，masks 应为 long 类型
#             masks = torch.tensor(masks, dtype=torch.long)
#
#         return self.criterion(outs, masks)
#
#     def forward(self, prediction, ground_truth):
#         encode_tensor = encode_mask(ground_truth)
#         masks_pred_sigmoid = torch.sigmoid(prediction)
#         # 二元交叉熵损失(逐像素)
#         loss_base = self.loss_ce(prediction, ground_truth)
#         # 血管密度损失
#         Loss_vd = torch.abs(masks_pred_sigmoid.sum() - encode_tensor.sum()) / (
#                 masks_pred_sigmoid.shape[0] * masks_pred_sigmoid.shape[2] * masks_pred_sigmoid.shape[3])
#
#
#         # 特征分布损失(全局)
#         masks_combined = torch.cat((masks_pred_sigmoid, encode_tensor), dim=1)  # Combine predicted and ground truth
#         counts = self.get_count(self.sizes, self.p, masks_combined)
#         vessel_ = torch.sqrt(torch.sum(self.sizes * (counts[..., 0] ** 2)))  # Vessel counts
#         size_t = torch.sqrt(torch.sum(self.sizes ** 2))
#         loss_FD = vessel_ / size_t / masks_combined.shape[0]
#
#
#
#         # 最终损失值
#         loss_value = self.beta * loss_base + self.gamma * Loss_vd + self.alpha * loss_FD
#         return loss_value



def encode_mask(ground_truth):
    # 将ground_truth转换为二分类（0:背景, 1:血管）
    encode_tensor = ground_truth.to(torch.float32)  # 转为float32类型
    encode_tensor = encode_tensor.to(device=torch.device('cuda'))  # 将数据移至GPU
    return encode_tensor


class CF_Loss(nn.Module):
    def __init__(self, beta=1.1, alpha=0.03, gamma=0.02, selected_mode=1):
        super(CF_Loss, self).__init__()
        self.beta = beta  # Binary Cross-Entropy Loss的权重
        self.alpha = alpha  # 特征分布损失的权重
        self.gamma = gamma  # 血管密度损失的权重
        self.sizes1 = [144, 72, 36]  # 所有要切割的尺寸  DCA1
        # self.sizes1 = [256, 128, 64]  # 所有要切割的尺寸   DRIVE
        # self.sizes1 = [480, 240, 160]  # 所有要切割的尺寸   CHASEDB1

        img_size = 288
        # img_size = 960

        self.p = torch.tensor(img_size, dtype=torch.float)
        self.n = torch.log(self.p) / torch.log(torch.tensor([2]).to('cuda'))
        self.n = torch.floor(self.n)
        self.sizes = 2 ** torch.arange(self.n.item(), 1, -1).to(dtype=torch.int)

        # self.sizes = torch.tensor([256, 128, 64, 32, 16, 8, 4, 2], dtype=torch.int)

        # 选取损失函数类型：1 为 BCELoss，2 为 CrossEntropyLoss
        self.selected_mode = selected_mode
        if self.selected_mode == 1:
            self.criterion = nn.BCELoss()
        elif self.selected_mode == 2:
            self.criterion = nn.CrossEntropyLoss()


# 原创新另一个服务器适用，这个服务器不适用，因此修改适配的损失函数----（16G）
#     def calculate_patch_loss(self, prediction, ground_truth):
#         total_loss = 0.0  # 初始化总损失变量
#         epsilon = 1e-6  # 避免除以0的风险
#
#         for patch_size in self.sizes1:
#             local_loss = 0.0  # 每次新的 patch_size 重置局部损失
#             total_sum = 0.0  # 对于每个patch_size重置总和变量
#             N, C, H, W = prediction.shape
#             num_patches = (H // patch_size) ** 2
#
#             for i in range(H // patch_size):
#                 for j in range(W // patch_size):
#                     start_i, start_j = i * patch_size, j * patch_size
#                     end_i, end_j = start_i + patch_size, start_j + patch_size
#
#                     pred_patch = prediction[:, :, start_i:end_i, start_j:end_j]
#                     gt_patch = ground_truth[:, :, start_i:end_i, start_j:end_j]
#
#                     # 将预测和真实标签拼接
#                     masks_combined = torch.cat((pred_patch, gt_patch), dim=1)
#
#                     # 计算 counts 背景为第0通道，血管为第1通道
#                     assert torch.all((masks_combined >= 0) & (masks_combined <= 1)), "Values in S are out of bounds!"
#
#                     counts = torch.zeros((masks_combined.shape[0], 1, 1))
#                     S = masks_combined
#
#                     # 计算 counts
#                     counts[..., 0, 0] = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / (
#                         torch.clamp((S[:, 1, ...] > 0).sum(), min=epsilon))
#
#                     # 累加当前 patch 的损失
#                     total_sum += torch.sum(patch_size * (counts[..., 0, 0] ** 2))
#
#             # 计算该 patch_size 下的损失，并累加到 total_loss
#             # total_sum = torch.sqrt(total_sum)
#             size_t = torch.sqrt(torch.tensor(num_patches * (patch_size ** 2), dtype=torch.float))
#             local_loss += total_sum / size_t / masks_combined.shape[0]
#
#             # 对局部损失进行梯度裁剪，防止局部损失过大
#             local_loss = torch.clamp(local_loss, min=0.0, max=1.0)
#
#             # 累加局部损失到总损失
#             total_loss += local_loss
#
#         return total_loss

# 原创新另一个服务器适用，这个服务器不适用，因此修改适配的损失函数----（24G）
    def calculate_patch_loss(self, prediction, ground_truth):
        total_loss = 0.0  # 初始化总损失变量
        epsilon = 1e-6  # 设置一个小常数，避免除以0的风险
        total_sum = 0.0  # 初始化当前 patch 总损失为 0

        for patch_size in self.sizes1:  # 遍历不同的 patch 大小
            local_loss = 0.0  # 每次新的 patch_size 重置局部损失
            total_sum = 0.0  # 对于每个 patch_size 重置总和变量
            N, C, H, W = prediction.shape  # 获取预测图像的形状 (批量大小, 通道数, 高度, 宽度)
            num_patches = (H // patch_size) ** 2  # 计算每个 patch 中的 patch 数量

            for i in range(H // patch_size):  # 遍历每个 patch 的行
                for j in range(W // patch_size):  # 遍历每个 patch 的列
                    # 计算当前 patch 的起始和结束位置
                    start_i, start_j = i * patch_size, j * patch_size
                    end_i, end_j = start_i + patch_size, start_j + patch_size

                    # 提取预测图像和 ground_truth 图像的当前 patch 区域
                    pred_patch = prediction[:, :, start_i:end_i, start_j:end_j]
                    gt_patch = ground_truth[:, :, start_i:end_i, start_j:end_j]

                    # 将预测和真实标签拼接在一起
                    masks_combined = torch.cat((pred_patch, gt_patch), dim=1)

                    # 计算 counts，背景为第 0 通道，血管为第 1 通道
                    assert torch.all((masks_combined >= 0) & (masks_combined <= 1)), "Values in S are out of bounds!"

                    # 计算 counts（血管区域与背景区域的差异）
                    S = masks_combined
                    non_zero_mask = (S[:, 1, ...] > 0).sum()  # 计算血管区域的数量

                    # 如果没有血管区域，则 counts 为零张量
                    if non_zero_mask == 0:
                        counts = torch.zeros((S.shape[0], 1, 1), device=S.device)  # 创建零张量
                    else:
                        # 否则，计算 counts（血管区域与背景区域的绝对差值的总和）
                        counts = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / torch.clamp(non_zero_mask, min=epsilon)

                    # 如果 counts 是标量或一维张量，扩展维度
                    if counts.ndimension() == 0:
                        counts = counts.unsqueeze(0).unsqueeze(0)  # 将 counts 扩展为一个伪二维张量
                    elif counts.ndimension() == 1:
                        counts = counts.unsqueeze(1)  # 扩展维度

                    # 累加当前 patch 的损失
                    total_sum += torch.sum(patch_size * (counts ** 2))  # 计算并累加当前 patch 的损失

            # 计算该 patch_size 下的损失，并累加到 total_loss
            size_t = torch.sqrt(torch.tensor(num_patches * (patch_size ** 2), dtype=torch.float))  # 计算当前 patch_size 对应的大小
            local_loss = local_loss + total_sum / size_t / masks_combined.shape[0]  # 计算局部损失

            # 对局部损失进行处理，防止过大
            local_loss = torch.clamp(local_loss, min=0.0, max=1.0)  # 限制损失在 [0, 1] 范围内

            # 累加局部损失到总损失
            total_loss = total_loss + local_loss

        return total_loss  # 返回总损失

# 原创新另一个服务器适用，这个服务器不适用，因此修改适配的损失函数-----（16G）

# 原创新另一个服务器适用，这个服务器不适用，因此修改适配的损失函数----（16G）
#     def get_count(self, sizes, p, masks_pred_sigmoid):
#         # 确保 counts 和其他张量都在 GPU 上，与 masks_pred_sigmoid 在同一设备上
#         counts = torch.zeros((masks_pred_sigmoid.shape[0], len(sizes), 1),
#                              device=masks_pred_sigmoid.device)  # 初始化 counts
#
#         epsilon = 1e-6  # 避免除以0的风险
#
#         for idx, size in enumerate(sizes):  # 使用 idx 来定位每个 patch size
#             stride = (size, size)
#
#             # 确保 pad_size 在正确的设备上
#             pad_size = torch.where((p % size) == 0, torch.tensor(0, dtype=torch.int, device=masks_pred_sigmoid.device),
#                                    (size - p % size).to(dtype=torch.int, device=masks_pred_sigmoid.device))
#
#             pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
#             pool = nn.AvgPool2d(kernel_size=(size, size), stride=stride)
#
#             # 确保 S 在正确的设备上
#             S = pad(masks_pred_sigmoid)
#             S = pool(S)
#             S = S * ((S > 0) & (S < (size * size)))  # 忽略无效区域
#
#             counts[..., 0, 0] = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / (
#                         torch.clamp((S[:, 1, ...] > 0).sum(), min=epsilon))
#
#         return counts

#原创新另一个服务器适用，这个服务器不适用，因此修改适配的损失函数----（24G）
    def get_count(self, sizes, p, masks_pred_sigmoid):
        # 确保 counts 和其他张量都在 GPU 上，与 masks_pred_sigmoid 在同一设备上
        counts = torch.zeros((masks_pred_sigmoid.shape[0], len(sizes), 1),
                             device=masks_pred_sigmoid.device)  # 初始化 counts

        epsilon = 1e-6  # 避免除以0的风险

        for idx, size in enumerate(sizes):  # 使用 idx 来定位每个 patch size
            stride = (size, size)

            # 确保 pad_size 在正确的设备上
            pad_size = torch.where((p % size) == 0, torch.tensor(0, dtype=torch.int, device=masks_pred_sigmoid.device),
                                   (size - p % size).to(dtype=torch.int, device=masks_pred_sigmoid.device))

            pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
            pool = nn.AvgPool2d(kernel_size=(size, size), stride=stride)

            # 确保 S 在正确的设备上
            S = pad(masks_pred_sigmoid)
            S = pool(S)
            S = S * ((S > 0) & (S < (size * size)))  # 忽略无效区域

            # 计算血管区域的数量
            non_zero_mask = (S[:, 1, ...] > 0).sum()  # 计算血管区域的数量

            # 如果没有血管区域，则 counts 为零张量
            if non_zero_mask == 0:
                counts[:, idx, 0] = torch.zeros(S.shape[0], device=S.device)  # 将对应的 counts 设置为零
            else:
                # 否则，计算 counts（血管区域与背景区域的绝对差值的总和）
                counts[:, idx, 0] = (S[:, 0, ...] - S[:, 1, ...]).abs().sum() / torch.clamp(non_zero_mask, min=epsilon)

        return counts

    def loss_ce(self, preds, masks):
        '''
        :param preds:       N  H  W  C (模型预测)
        :param masks:       N  1  H  W (真实标签)
        :return:            返回根据选定模式计算的损失值
        '''
        if self.selected_mode == 1:  # 使用 BCELoss 时，masks 应为 float 类型
            masks = torch.tensor(masks, dtype=torch.float)

        outs = preds.permute((0, 2, 3, 1))  # 调整预测输出形状为 N H W C
        outs = outs.reshape((-1, outs.size()[3]))  # 调整为 (N*H*W, C)

        if self.selected_mode == 1:
            outs = outs.reshape((-1,))  # BCELoss 需要展平成单通道
        masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1

        if self.selected_mode == 2:  # 使用 CrossEntropyLoss 时，masks 应为 long 类型
            masks = torch.tensor(masks, dtype=torch.long)

        return self.criterion(outs, masks)

    def forward(self, prediction, ground_truth):
        encode_tensor = encode_mask(ground_truth)
        masks_pred_sigmoid = torch.sigmoid(prediction)
        # 二元交叉熵损失(逐像素)
        loss_base = self.loss_ce(prediction, ground_truth)
        # # 血管密度损失
        # Loss_vd = torch.abs(masks_pred_sigmoid.sum() - encode_tensor.sum()) / (
        #         masks_pred_sigmoid.shape[0] * masks_pred_sigmoid.shape[2] * masks_pred_sigmoid.shape[3])


        # 特征分布损失(全局)
        # Ensure that masks_pred_sigmoid and encode_tensor are on the same device
        masks_combined = torch.cat((masks_pred_sigmoid, encode_tensor), dim=1)  # Combine predicted and ground truth
        # Ensure self.sizes is on the same device as masks_combined
        self.sizes = torch.tensor(self.sizes, device=masks_combined.device)
        # Call get_count with masks_combined on the correct device
        counts = self.get_count(self.sizes, self.p, masks_combined)
        # Vessel counts
        vessel_ = torch.sqrt(torch.sum(self.sizes * (counts[..., 0] ** 2)))  # Vessel counts
        # Size normalization
        size_t = torch.sqrt(torch.sum(self.sizes ** 2))
        # Feature distribution loss
        loss_FD = vessel_ / size_t / masks_combined.shape[0]

        # loss_FD = torch.clamp(loss_FD, min=0.0, max=1.0) # FIVES  数据集操作变了=======================================================================================================

        # 特征分布损失(从局部到全局)
        total_loss_fd_gamma = 0.0
        total_loss_fd_gamma = self.calculate_patch_loss(masks_pred_sigmoid, encode_tensor)

        # 最终损失值
        loss_value = self.beta * loss_base + self.gamma * loss_FD + self.alpha * total_loss_fd_gamma
        return loss_value



def loss_ce_ds(preds, masks, criterion, selected_mode = 2):
    # this is used to calculate cross-entropy with many categories !
    if selected_mode ==1:       # when choose nn.BCELoss() masks should be float !
        masks = torch.tensor(masks, dtype=torch.float)
    outs0 = preds[0].permute((0, 2, 3, 1))  # N H W C
    outs0 = outs0.reshape((-1, outs0.size()[3]))  # N*H*W, C
    outs1 = preds[1].permute((0, 2, 3, 1))  # N H W C
    outs1 = outs1.reshape((-1, outs1.size()[3]))  # N*H*W, C
    outs2 = preds[2].permute((0, 2, 3, 1))  # N H W C
    outs2 = outs2.reshape((-1, outs2.size()[3]))  # N*H*W, C
    outs3 = preds[3].permute((0, 2, 3, 1))  # N H W C
    outs3 = outs3.reshape((-1, outs3.size()[3]))  # N*H*W, C
    masks = masks.reshape((-1,))  # N,1,H,W ===> N,-1
    masks = torch.tensor(masks, dtype=torch.long)
    loss = 0.25 * criterion(outs0, masks) + 0.5 * criterion(outs1, masks) + \
           0.75 * criterion(outs2, masks) + 1.0 * criterion(outs3, masks)
    return loss

if __name__ == '__main__':
    labels = torch.tensor([0, 1, 1, 0, 1, 1])
    pred = torch.tensor([[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.4, 0.6], [0.3, 0.7], [0.3, 0.7]])
    pred2 = torch.tensor([0.3, 0.7, 0.6, 0.2, 0.5, 0.9])

    print(weighted_entropy(need_soft_max = False)(pred,labels))
    print(DiceLoss()(pred2, labels))
    print(FocalLoss()(pred2, torch.tensor(labels, dtype=torch.float)))
    print(nn.CrossEntropyLoss()(pred, labels))

    pass