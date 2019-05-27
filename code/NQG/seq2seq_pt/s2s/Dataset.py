from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import s2s


class Dataset(object):
    def __init__(self, srcData, bioData, featsData, tgtData, copySwitchData, copyTgtData,
                 ansData, ansfeatsData, batchSize, cuda, copy, answer, feature, answer_feature):
        self.src = srcData
        self.ans = ansData
        self.ansfeats = ansfeatsData
        self.answer = answer
        self.bio = bioData
        self.feature = feature
        self.answer_feature = answer_feature
        self.feats = featsData
        self.copy = copy
        self.copySwitch = None
        self.copyTgt = None
        if tgtData:
            self.tgt = tgtData
            if copy:
                # copy switch should company tgt label
                self.copySwitch = copySwitchData
                self.copyTgt = copyTgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        bioBatch, featBatches, ansBatch, ans_length, ansfeatBatches = None, None, None, None, None
        if self.answer == 'encoder':
            ansBatch, ans_length = self._batchify(
                self.ans[index * self.batchSize:(index + 1) * self.batchSize],
                align_right=False, include_lengths=True)
            if self.answer_feature:
                ansfeatBatches = [self._batchify(x[index * self.batchSize:(index + 1) * self.batchSize], align_right=False) for x in zip(*self.ansfeats)]
        elif self.answer == 'embedding':
            bioBatch = self._batchify(self.bio[index * self.batchSize:(index + 1) * self.batchSize], align_right=False)
        if self.feature:
            featBatches = [self._batchify(x[index * self.batchSize:(index + 1) * self.batchSize], align_right=False) for x
                       in zip(*self.feats)]

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])

        else:
            tgtBatch = None

        if self.copySwitch is not None:
            copySwitchBatch = self._batchify(
                self.copySwitch[index * self.batchSize:(index + 1) * self.batchSize])
            copyTgtBatch = self._batchify(
                self.copyTgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            copySwitchBatch = None
            copyTgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, bioBatch, *featBatches)
        else:
            if bioBatch is not None:
                if self.copySwitch is not None:
                    if featBatches is not None:
                        batch = zip(indices, srcBatch, bioBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch)
                    else:
                        batch = zip(indices, srcBatch, bioBatch, tgtBatch, copySwitchBatch, copyTgtBatch)
                else:
                    if featBatches is not None:
                        batch = zip(indices, srcBatch, bioBatch, *featBatches, tgtBatch)
                    else:
                        batch = zip(indices, srcBatch, bioBatch, tgtBatch)
            elif ansBatch is not None:
                if ansfeatBatches is not None:
                    if self.copySwitch is not None:
                        if featBatches is not None:
                            batch = zip(indices, srcBatch, ansBatch, *ansfeatBatches, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch)
                        else:
                            batch = zip(indices, srcBatch, ansBatch, *ansfeatBatches, tgtBatch, copySwitchBatch, copyTgtBatch)
                    else:
                        if featBatches is not None:
                            batch = zip(indices, srcBatch, ansBatch, *ansfeatBatches, *featBatches, tgtBatch)
                        else:
                            batch = zip(indices, srcBatch, ansBatch, *ansfeatBatches, tgtBatch)
                else:
                    if self.copySwitch is not None:
                        if featBatches is not None:
                            batch = zip(indices, srcBatch, ansBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch)
                        else:
                            batch = zip(indices, srcBatch, ansBatch, tgtBatch, copySwitchBatch, copyTgtBatch)
                    else:
                        if featBatches is not None:
                            batch = zip(indices, srcBatch, ansBatch, *featBatches, tgtBatch)
                        else:
                            batch = zip(indices, srcBatch, ansBatch, tgtBatch)
            else:
                if self.copySwitch is not None:
                    if featBatches is not None:
                        batch = zip(indices, srcBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch)
                    else:
                        batch = zip(indices, srcBatch, tgtBatch, copySwitchBatch, copyTgtBatch)
                else:
                    if featBatches is not None:
                        batch = zip(indices, srcBatch, *featBatches, tgtBatch)
                    else:
                        batch = zip(indices, srcBatch, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        if self.answer == 'encoder':
            batch, lengths, ans_length = zip(*sorted(zip(batch, lengths, ans_length), key=lambda x: -x[1]))
        else:
            batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            if bioBatch is not None:
                if featBatches is not None:
                    indices, srcBatch, bioBatch, *featBatches = zip(*batch)
                else:
                    indices, srcBatch, bioBatch = zip(*batch)
            else:
                if featBatches is not None:
                    indices, bioBatch, *featBatches = zip(*batch)
                else:
                    indices, srcBatch = zip(*batch)
        else:
            if self.copySwitch is not None:
                if bioBatch is not None:
                    if featBatches is not None:
                        indices, srcBatch, bioBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                    else:
                        indices, srcBatch, bioBatch, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                elif ansBatch is not None:
                    if ansfeatBatches is not None:
                        if featBatches is not None:
                            indices, srcBatch, ansBatch, *ansfeatBatches, f1, f2, f3, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                            featBatches = [f1, f2, f3]
                        else:
                            indices, srcBatch, ansBatch, *ansfeatBatches, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                    else:
                        if featBatches is not None:
                            indices, srcBatch, ansBatch, f1, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                            featBatches = [f1]
                        else:
                            indices, srcBatch, ansBatch, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                else:
                    if featBatches is not None:
                        indices, srcBatch, *featBatches, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
                    else:
                        indices, srcBatch, tgtBatch, copySwitchBatch, copyTgtBatch = zip(*batch)
            else:
                if bioBatch is not None:
                    if featBatches is not None:
                        indices, srcBatch, bioBatch, *featBatches, tgtBatch = zip(*batch)
                    else:
                        indices, srcBatch, bioBatch, tgtBatch = zip(*batch)
                elif ansBatch is not None:
                    if ansfeatBatches is not None:
                        if featBatches is not None:
                            indices, srcBatch, ansBatch, *ansfeatBatches, f1, f2, f3, tgtBatch = zip(*batch)
                            featBatches = [f1, f2, f3]
                        else:
                            indices, srcBatch, ansBatch, *ansfeatBatches, tgtBatch = zip(*batch)
                    else:
                        if featBatches is not None:
                            indices, srcBatch, ansBatch, *featBatches, tgtBatch = zip(*batch)
                        else:
                            indices, srcBatch, ansBatch, tgtBatch = zip(*batch)
                else:
                    if featBatches is not None:
                        indices, srcBatch, *featBatches, tgtBatch = zip(*batch)
                    else:
                        indices, srcBatch, tgtBatch = zip(*batch)

        if featBatches is not None:
            featBatches = list(featBatches)
        if ansfeatBatches is not None:
            ansfeatBatches = list(ansfeatBatches)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        if self.answer == 'encoder':
            ans_length = torch.LongTensor(ans_length).view(1, -1)

        if featBatches is not None:
            if self.answer == 'encoder':
                if ansfeatBatches is not None:
                    return (wrap(srcBatch), lengths), \
                           (wrap(ansBatch), ans_length), (tuple(wrap(x) for x in featBatches), lengths), \
                           (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                           (tuple(wrap(x) for x in ansfeatBatches), ans_length), \
                           indices
                return (wrap(srcBatch), lengths), \
                       (wrap(ansBatch), ans_length), (tuple(wrap(x) for x in featBatches), lengths), \
                       (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                       indices
            else:
                return (wrap(srcBatch), lengths), \
                       (wrap(bioBatch), lengths), (tuple(wrap(x) for x in featBatches), lengths), \
                       (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                       indices
        else:
            if self.answer == 'encoder':
                if ansfeatBatches is not None:
                    return (wrap(srcBatch), lengths), \
                           (wrap(ansBatch), ans_length), \
                           (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                           (tuple(wrap(x) for x in ansfeatBatches), ans_length), \
                           indices
                return (wrap(srcBatch), lengths), \
                       (wrap(ansBatch), ans_length), \
                       (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                       indices
            else:
                return (wrap(srcBatch), lengths), \
                       (wrap(bioBatch), lengths), \
                       (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                       indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        if self.copy:
            if self.feature:
                if self.answer == 'embedding':
                    data = list(
                        zip(self.src, self.bio, self.feats, self.tgt, self.copySwitch, self.copyTgt))
                    self.src, self.bio, self.feats, self.tgt, self.copySwitch, self.copyTgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
                elif self.answer == 'encoder':
                    if self.answer_feature:
                        data = list(
                            zip(self.src, self.ans, self.ansfeats, self.feats, self.tgt, self.copySwitch, self.copyTgt))
                        self.src, self.ans, self.ansfeats, self.feats, self.tgt, self.copySwitch, self.copyTgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                    else:
                        data = list(
                            zip(self.src, self.ans, self.feats, self.tgt, self.copySwitch, self.copyTgt))
                        self.src, self.ans, self.feats, self.tgt, self.copySwitch, self.copyTgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                else:
                    data = list(
                        zip(self.src, self.feats, self.tgt, self.copySwitch, self.copyTgt))
                    self.src, self.feats, self.tgt, self.copySwitch, self.copyTgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
            else:
                if self.answer == 'embedding':
                    data = list(
                        zip(self.src, self.bio, self.tgt, self.copySwitch, self.copyTgt))
                    self.src, self.bio, self.tgt, self.copySwitch, self.copyTgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
                elif self.answer == 'encoder':
                    if self.answer_feature:
                        data = list(
                            zip(self.src, self.ans, self.ansfeats, self.tgt, self.copySwitch, self.copyTgt))
                        self.src, self.ans, self.ansfeats, self.tgt, self.copySwitch, self.copyTgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                    else:
                        data = list(
                            zip(self.src, self.ans, self.tgt, self.copySwitch, self.copyTgt))
                        self.src, self.ans, self.tgt, self.copySwitch, self.copyTgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                else:
                    data = list(
                        zip(self.src, self.tgt, self.copySwitch, self.copyTgt))
                    self.src, self.tgt, self.copySwitch, self.copyTgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
        else:
            if self.feature:
                if self.answer == 'embedding':
                    data = list(
                        zip(self.src, self.bio, self.feats, self.tgt))
                    self.src, self.bio, self.feats, self.tgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
                elif self.answer == 'encoder':
                    if self.answer_feature:
                        data = list(
                            zip(self.src, self.ans, self.ansfeats, self.feats, self.tgt))
                        self.src, self.ans, self.feats, self.tgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                    else:
                        data = list(
                            zip(self.src, self.ans, self.ansfeats, self.feats, self.tgt))
                        self.src, self.ans, self.feats, self.tgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                else:
                    data = list(
                        zip(self.src, self.feats, self.tgt))
                    self.src, self.feats, self.tgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
            else:
                if self.answer == 'embedding':
                    data = list(
                        zip(self.src, self.bio, self.tgt))
                    self.src, self.bio, self.tgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
                elif self.answer == 'encoder':
                    if self.answer_feature:
                        data = list(
                            zip(self.src, self.ans, self.ansfeats, self.tgt))
                        self.src, self.ans, self.ansfeats, self.tgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                    else:
                        data = list(
                            zip(self.src, self.ans, self.tgt))
                        self.src, self.ans, self.tgt = zip(
                            *[data[i] for i in torch.randperm(len(data))])
                else:
                    data = list(
                        zip(self.src, self.tgt))
                    self.src, self.tgt = zip(
                        *[data[i] for i in torch.randperm(len(data))])
