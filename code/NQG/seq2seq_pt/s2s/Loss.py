from __future__ import print_function
import math
import torch.nn as nn
import numpy as np
import torch

class Loss(object):

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        if not issubclass(type(self.criterion), nn.modules.loss._Loss):
            raise ValueError("Criterion has to be a subclass of torch.nn._Loss")
        # accumulated loss
        self.acc_loss = 0
        # normalization term
        self.norm_term = 0

    def reset(self):
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        raise NotImplementedError

    def cuda(self):
        self.criterion.cuda()

    def backward(self):
        if type(self.acc_loss) is int:
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()

class NLLLoss(Loss):

    _NAME = "NLLLoss"

    def __init__(self, weight=None, mask=None, size_average=True, copy_loss=False, coverage_loss=False, coverage_weight=None):
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super(NLLLoss, self).__init__(
            self._NAME,
            nn.NLLLoss(weight=weight, size_average=size_average))

        self.copy = copy_loss
        if copy_loss:
            self.copy_loss = nn.NLLLoss(size_average=False)
        
        self.coverage = coverage_loss
        if coverage_loss:
            self.coverage_weight = coverage_weight

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data.item()
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1

    def loss_function(self, g_outputs, g_targets, generator, c_outputs=None, c_switch=None,
                      c_targets=None, c_gate_values=None, coverage_outputs=None):
        batch_size = g_outputs.size(1)

        g_out_t = g_outputs.view(-1, g_outputs.size(2))
        g_prob_t = generator(g_out_t)
        g_prob_t = g_prob_t.view(-1, batch_size, g_prob_t.size(1))

        if self.copy:
            c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
            g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

            c_output_prob_log = torch.log(c_output_prob)
            g_output_prob_log = torch.log(g_output_prob)
            c_output_prob_log = c_output_prob_log * (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
            g_output_prob_log = g_output_prob_log * ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

            g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
            c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))

            g_loss = self.criterion(g_output_prob_log, g_targets.view(-1))
            c_loss = self.copy_loss(c_output_prob_log, c_targets.view(-1))
            total_loss = g_loss + c_loss
        else:
            g_prob_t_log = torch.log(g_prob_t)
            g_prob_t_log = g_prob_t_log.view(-1, g_prob_t_log.size(2))
            g_loss = self.criterion(g_prob_t_log, g_targets.view(-1))
            total_loss = g_loss
        
        if self.coverage:
            coverage_outputs = [co for co in coverage_outputs]
            coverage_loss = torch.sum(torch.stack(coverage_outputs, 1), 1)
            coverage_loss = torch.sum(coverage_loss, 0)
            report_loss = total_loss.item()
            total_loss = total_loss + coverage_loss * self.coverage_weight
        else:
            report_loss = total_loss.item()
        return total_loss, report_loss, 0
