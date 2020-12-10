import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):  # python3.x has no type of long
#             self.alpha = torch.tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, output, target):
#         if output.dim() > 2:
#             output = output.flatten(start_dim=2)
#             output = output.transpose(1, 2)
#             output = output.contiguous().view(-1, output.size(2))
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(output, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != output.data.type():
#                 self.alpha = self.alpha.type_as(output.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             lpgpt = logpt * Variable(at)
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


'''
.data vs .detach()
Any in-place change on x.detach() will cause errors when x is needed in backward, 
so .detach() is a safer way for the exclusion of subgraphs from gradient computation.
'''


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            loss = F.nll_loss(logpt, target, torch.tensor(self.alpha).to(target.device), ignore_index=self.ignore_index)
        else:
            loss = F.nll_loss(logpt, target, ignore_index=self.ignore_index)
        return loss

# class FocalLoss(nn.Module):
#     """
#     This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#     'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#         Focal_Loss= -1*alpha*(1-pt)*log(pt)
#     :param num_class:
#     :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#     :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                     focus on hard misclassified example
#     :param smooth: (float,double) smooth value when cross entropy
#     :param balance_index: (int) balance class index, should be specific when alpha is float
#     :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#     """
#
#     def __init__(self, num_class, gamma=2, alpha=None, balance_index=-1, smooth=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.num_class = num_class
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.size_average = size_average
#
#         if self.alpha is None:
#             self.alpha = torch.ones(self.num_class, 1)
#         elif isinstance(self.alpha, (list, np.ndarray)):
#             assert len(self.alpha) == self.num_class
#             self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
#             self.alpha = self.alpha / self.alpha.sum()
#         elif isinstance(self.alpha, float):
#             alpha = torch.ones(self.num_class, 1)
#             alpha = alpha * (1 - self.alpha)
#             alpha[balance_index] = self.alpha
#             self.alpha = alpha
#         else:
#             raise TypeError('Not support alpha type')
#
#         if self.smooth is not None:
#             if self.smooth < 0 or self.smooth > 1.0:
#                 raise ValueError('smooth value should be in [0,1]')
#
#     def forward(self, logit, target):
#
#         if logit.dim() > 2:
#             # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#             logit = logit.view(logit.size(0), logit.size(1), -1)
#             logit = logit.permute(0, 2, 1).contiguous()
#             logit = logit.view(-1, logit.size(-1))
#         target = target.view(-1, 1)
#
#         epsilon = 1e-10
#         alpha = self.alpha
#         if alpha.device != logit.device:
#             alpha = alpha.to(logit.device)
#
#         idx = target.cpu().long()
#
#         one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
#         one_hot_key = one_hot_key.scatter_(1, idx, 1)
#         if one_hot_key.device != logit.device:
#             one_hot_key = one_hot_key.to(logit.device)
#
#         if self.smooth:
#             one_hot_key = torch.clamp(
#                 one_hot_key, self.smooth/(self.num_class-1), 1.0 - self.smooth)
#         pt = (one_hot_key * logit).sum(1) + epsilon
#         logpt = pt.log()
#
#         gamma = self.gamma
#
#         alpha = alpha[idx]
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
#
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
