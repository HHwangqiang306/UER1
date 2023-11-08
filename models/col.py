# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from utils.losses_negative_only import SupConLoss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # GIL Param
    """
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')
    parser.add_argument('--tiny_imgnet_path', type=str, default='data')
    """
    return parser


class COL(ContinualModel):
    NAME = 'col'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(COL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.criterion = SupConLoss()
        #self.stable_model = deepcopy(self.net).to(self.device)
        self.current_task = 0
        self.global_step = 0
        self.current_temp = 0.07
        self.past_temp = 0.07
        self.distill_power = 0.1

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        rot_inputs = not_aug_inputs.cpu()
        for i in range(0, not_aug_inputs.shape[0]):
            #rot_inputs[i] = self.transform(rot_inputs[i])
            rot_inputs[i] = torch.rot90(rot_inputs[i], 2, [-1, -2])#(rot_inputs[i])
        rot_inputs = rot_inputs.to(self.device)

        if not self.buffer.is_empty():
            buf_inputs, buf_aug, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, multiple_aug=True)
            inputs = torch.cat((inputs, buf_inputs))
            rot_inputs = torch.cat((rot_inputs, buf_aug))
            labels = torch.cat((labels, buf_labels))
        bsz = inputs.shape[0]
        outputs, features = self.net(inputs, return_features = True)
        loss = self.loss(outputs, labels)
        # Asym SupCon
        _, aug_features = self.net(rot_inputs, return_features=True)
        #f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        feature = torch.cat([features.unsqueeze(1), aug_features.unsqueeze(1)], dim=1)
        loss += 0.3*self.criterion(feature, labels)

        # IRD (current)
        if hasattr(self, 'ref_model'):
            features1_prev_task = features

            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), self.current_temp)
            logits_mask = torch.scatter(torch.ones_like(features1_sim),1,
                torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True), 0)
            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
            features1_sim = features1_sim - logits_max1.detach()
            row_size = features1_sim.size(0)
            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


        # IRD (past)
            with torch.no_grad():
                features2_prev_task = self.ref_model(inputs)
                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), self.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += self.distill_power * loss_distill
            #distill.update(loss_distill.item(), bsz)
        """"""

        loss.backward()
        self.opt.step()

        self.global_step += 1
        """
        if self.global_step % self.step ==0:
            self.stable_model = deepcopy(self.net).to(self.device)
        """
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:not_aug_inputs.shape[0]])

        return loss.item()


    def end_task(self, dataset) -> None:
        self.ref_model = deepcopy(self.net).to(self.device)

