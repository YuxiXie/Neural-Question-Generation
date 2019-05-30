import torch
import torch.nn as nn
import math

try:
    import ipdb
except ImportError:
    pass


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim, is_coverage=False):
        super(ConcatAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None
        self.is_coverage = is_coverage
        if is_coverage:
            self.linear_cov = nn.Linear(1, att_dim, bias=False)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None, coverage=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim

        if self.is_coverage:
            coverage_input = coverage.view(-1, 1)  # (batch x sourceL) x 1
            weighted_coverage = self.linear_cov(coverage_input).view(-1, tmp10.size(1), self.att_dim)  # batch x sourceL x att_dim
            tmp10 = tmp10 + weighted_coverage

        tmp20 = self.tanh(tmp10)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL
        if self.mask is not None:
            # energy.data.masked_fill_(self.mask, -float('inf'))
            # energy.masked_fill_(self.mask, -float('inf'))   # TODO: might be wrong
            energy = energy * (1 - self.mask) + self.mask * (-1000000)
        
        score_pre = self.sm(energy)  # batch x sourceL
        normalization_term = score_pre.sum(1, keepdim=True)
        score = score_pre / normalization_term  # batch x sourceL
        
        score_m = score.unsqueeze(1)  # batch x 1 x sourceL
        weightedContext = torch.bmm(score_m, context).squeeze(1)  # batch x dim

        if self.is_coverage:
            coverage = coverage.view(-1, score.size(1))
            coverage = coverage + score  # batch x sourceL

            return weightedContext, score, precompute, coverage
        
        return weightedContext, score, precompute

    def extra_repr(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
