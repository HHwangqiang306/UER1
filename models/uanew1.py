# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch.nn as nn
import torch
import torch.nn.functional as F
from datasets import get_dataset
from utils.buffer_UA1 import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
from math import log
import random, torch, time, logging
from backbone.BNN2 import BayesianCNN
import torchvision.transforms as transforms
from utils.scloss import SupConLoss
from PIL import Image
from torchstat import stat



def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' uanew.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--T_NUMBER', type=float, required=True,
                        help='forward_stochastic number.')
    parser.add_argument('--M', type=float, required=True,
                        help='forward_stochastic number.')


    return parser


def rotate_img(img, s):
    transform = transforms.RandomResizedCrop(size=(32, 32), scale=(0.66, 0.67), ratio=(0.99, 1.00))
    # image = Image.fromarray(img)
    img = transform(img)
    return torch.rot90(img, s, [-1, -2])
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
kdloss = KDLoss(3.5)


class Uanew1(ContinualModel):
    NAME = 'uanew1'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Uanew1, self).__init__(backbone, loss, args, transform)
        self.opt.zero_grad()
        self.model2 = BayesianCNN().to(self.device)
        self.criterion = SupConLoss()
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        # print(inputs)
        outputs1, _, bat_inv = self.net(inputs, return_features=True)
        #model=model2.net
        # outputs2 = self.net(not_aug_inputs)

        T = self.args.T_NUMBER
        M = int(self.args.M)
        outputs3 = self.model2.forward_stochastic(not_aug_inputs, k=M).mean(dim=-1)
        #loss=F.mse_loss(outputs1,outputs3)
        loss1 = self.loss(outputs1, labels, reduction='none')
        # loss2 = self.loss(outputs2, labels, reduction='none')
        loss3 = self.loss(outputs3, labels, reduction='none')
        # loss3 = loss31.cuda().data.cpu().numpy()
        loss_scores = T*loss1 +(1-T)*loss3
        # print(loss_scores)
        loss = loss_scores.mean()
        with torch.no_grad():
            probs = outputs3.exp()
            probs = probs.unsqueeze(1)
            p_yc = probs.mean(dim=-1)
            # compute entropy and sum over class dimension (giving total uncertainty)
            H_y1 = - (p_yc * p_yc.log()).sum(dim=-1)
            # compute aleatoric uncertainty
            E_H_y1 = -(probs * probs.log()).sum(dim=-1).mean(dim=-1)
            # print(E_H_y1, E_H_y1.size())
            # deduce epistemic uncertainty
            BALD_acq1 = H_y1 - E_H_y1
            var_ratios2 = 1 - p_yc.max(dim=-1).values.detach().cpu().numpy()
            var_ratios = torch.tensor(var_ratios2).to(self.device)
            BALD_acq1 = (1 - BALD_acq1) * var_ratios

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, _, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _, _, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_outputs, _, buf_inv = self.net(buf_inputs, return_features=True)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            bat_inv = torch.cat((bat_inv, buf_inv))

            buf_inputs, _, _, mean_stds, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_indexes=True)
            buf_probs = self.model2.forward_stochastic(buf_inputs, k=M).exp()
            # buf_probs = buf_probs.unsqueeze(1)
            buf_E_H_y1 = -(buf_probs * buf_probs.log()).sum(dim=1).mean(dim=-1)
            # print(buf_E_H_y1,buf_E_H_y1.size())
            # exit()
            # mean_stds1 = torch.tensor(mean_stds1).to(self.device)
            loss += self.args.gamma * F.smooth_l1_loss(buf_E_H_y1, mean_stds)

            # _, buf_labels, buf_logits, buf_indexes = self.buffer.get_data(
            # self.args.minibatch_size, transform=self.transform, return_indexes=True)
            # oss += self.args.gamma * self.loss(buf_logits, buf_labels)
        label_shot = torch.arange(4).repeat(inputs.shape[0])
        label_shot = label_shot.type(torch.LongTensor)
        choice = np.random.choice(a=inputs.shape[0], size=inputs.shape[0], replace=False)
        rot_label = label_shot[choice].to(self.device)
        rot_inputs = inputs.cpu()
        for i in range(0, inputs.shape[0]):
            rot_inputs[i] = rotate_img(rot_inputs[i], rot_label[i])
        rot_inputs = rot_inputs.to(self.device)
        _, rot_outputs, t_inv = self.net(rot_inputs, return_features=True)
        loss += 0.3 * self.criterion(bat_inv, t_inv, labels)
        #loss += 0.3 * self.loss(rot_outputs, rot_label)

        loss.backward()
        self.opt.step()
        #stat(self.model2, inputs)
        #exit()
        #stat(self.net,(3,32,32))
        if not self.buffer.is_empty():
            self.buffer.update_scores(buf_indexes, BALD_acq1.detach())
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs1.data,
                             mean_stds=E_H_y1,
                             loss_scores=BALD_acq1.detach(),)
        return loss.item()
