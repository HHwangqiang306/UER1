# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import os
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import numpy as np
from utils.buffer import Buffer
from utils.triplet import merge
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from pytorch_metric_learning.miners import TripletMarginMiner

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--sigmoid', type=float, required=True,
                        help='Penalty weight.')
    return parser
miner = TripletMarginMiner(margin=0.2, type_of_triplets='semihard')
loss_function = nn.KLDivLoss(reduction='batchmean')
class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
class CC(nn.Module):

	def __init__(self, gamma, P_order):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order

	def forward(self, feat_s, feat_t):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		loss = F.mse_loss(corr_mat_s, corr_mat_t)

		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
kdloss = KDLoss(3.5)
closs = Correlation()
ccloss = CC(2, 3)
fcloss = focal_loss()

class Relation(ContinualModel):
    NAME = 'relation'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Relation, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)     #LC

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_rehearsal1 = ccloss(buf_outputs, buf_logits)+closs(buf_outputs, buf_logits)
            #L3
            loss += self.args.alpha * loss_rehearsal1

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss_rehearsal2 = self.loss(buf_outputs, buf_labels)    #ER
            loss += self.args.beta * loss_rehearsal2

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_embedding = F.normalize(buf_outputs, p=2, dim=1)
            anchor_id, positive_id, negative_id = miner(buf_embedding, buf_labels)
            anchor = buf_embedding[anchor_id]
            positive = buf_embedding[positive_id]
            negative = buf_embedding[negative_id]
            ap_dist = torch.norm(anchor - positive, p=2, dim=1)
            an_dist = torch.norm(anchor - negative, p=2, dim=1)
            loss_rehearsal4 = -torch.log(torch.exp(-ap_dist) / (torch.exp(-an_dist) + torch.exp(-ap_dist))).mean()
            #L2 distance
            loss += self.args.sigmoid * loss_rehearsal4

            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            teacher_embedding = F.normalize(buf_logits, p=2, dim=1)
            student_embedding = F.normalize(buf_outputs, p=2, dim=1)

            # if not self.args.flag_merge:
                # generate triplets
            with torch.no_grad():
                anchor_id, positive_id, negative_id = miner(student_embedding, buf_labels)

            # get teacher embedding in tuples
            teacher_anchor = teacher_embedding[anchor_id]
            teacher_positive = teacher_embedding[positive_id]
            teacher_negative = teacher_embedding[negative_id]
            # get student embedding in triplets
            student_anchor = student_embedding[anchor_id]
            student_positive = student_embedding[positive_id]
            student_negative = student_embedding[negative_id]
            # get a-p dist and a-n dist in teacher embedding
            teacher_ap_dist = torch.norm(teacher_anchor - teacher_positive, p=2, dim=1)
            teacher_an_dist = torch.norm(teacher_anchor - teacher_negative, p=2, dim=1)
            # get a-p dist and a-n dist in student embedding
            student_ap_dist = torch.norm(student_anchor - student_positive, p=2, dim=1)
            student_an_dist = torch.norm(student_anchor - student_negative, p=2, dim=1)
            # get probability of triplets in teacher embedding
            teacher_prob = torch.sigmoid((teacher_an_dist - teacher_ap_dist) / 4)
            teacher_prob_aug = torch.cat([teacher_prob.unsqueeze(1), 1 - teacher_prob.unsqueeze(1)])
            # get probability of triplets in student embedding
            student_prob = torch.sigmoid((student_an_dist - student_ap_dist) / 4)
            student_prob_aug = torch.cat([student_prob.unsqueeze(1), 1 - student_prob.unsqueeze(1)])
            # compute loss function
            loss_value = 1000 * loss_function(torch.log(student_prob_aug), teacher_prob_aug)

            loss_rehearsal3 = torch.mean(torch.sum(loss_value, dim=0))  #L2 distribution
            loss += self.args.gamma * loss_rehearsal3.cpu().item() * student_prob.size()[0]


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data
                             )

        return loss.item()
