from __future__ import division

import s2s
import torch
from torch import cuda
from nltk.translate import bleu_score
import os


class Evaluator(object):

    def __init__(self, opt, translator):
        self.evalModelCount = 0
        self.opt = opt
        self.translator = translator

    def get_bleu(self, batch, raw):
        if self.opt.answer == 'encoder':
            bio = None
            if self.opt.answer_feature:
                if self.opt.feature:
                    src, ans, feats, tgt, ansfeats, indices = batch[0]
                else:
                    src, ans, tgt, ansfeats, indices = batch[0]
                    feats = None
            else:
                ansfeats = None
                if self.opt.feature:
                    src, ans, feats, tgt, indices = batch[0]
                else:
                    src, ans, tgt, indices = batch[0]
                    feats = None
        else:
            ans, ansfeats = None, None
            if self.opt.feature:
                src, bio, feats, tgt, indices = batch[0]
            else:
                src, bio, tgt, indices = batch[0]
                feats = None
        src_batch, tgt_batch = raw

        #  translate
        pred, predScore, predIsCopy, predCopyPosition, attn, _ = self.translator.translateBatch(src, bio, feats, tgt, ans, ansfeats)
        if self.opt.copy:
            pred, predScore, predIsCopy, predCopyPosition, attn = list(zip(
                *sorted(zip(pred, predScore, predIsCopy, predCopyPosition, attn, indices),
                        key=lambda x: x[-1])))[:-1]
        else:
            pred, predScore, attn = list(zip(*sorted(zip(pred, predScore, attn, indices),
                        key=lambda x: x[-1])))[:-1]

        #  convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            n = 0
            if self.opt.copy:
                predBatch.append(
                    self.translator.buildTargetTokens(pred[b][n], src_batch[b],
                                                 predIsCopy[b][n], predCopyPosition[b][n], attn[b][n])
                )
            else:
                predBatch.append(
                    self.translator.buildTargetTokens(pred[b][n], src_batch[b], None, None, attn[b][n])
                )

        # nltk BLEU evaluator needs tokenized sentences
        gold = [[r] for r in tgt_batch]
        predict = predBatch
        no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '') for word in sent] for sent in predict]

        bleu = bleu_score.corpus_bleu(gold, no_copy_mark_predict)
        return bleu


    def evalModel(self, model, evalData):
        self.evalModelCount += 1
        ofn = 'dev.out.{0}'.format(self.evalModelCount)
        if self.opt.save_path:
            ofn = os.path.join(self.opt.save_path, ofn)

        predict, gold = [], []
        processed_data, raw_data = evalData

        report_loss, report_tgt_words = 0, 0

        for batch, raw_batch in zip(processed_data, raw_data):
            """
            (wrap(srcBatch), lengths), \
                   (wrap(bioBatch), lengths), (tuple(wrap(x) for x in featBatches), lengths), \
                   (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
                   indices
            """
            bio, feats, ans, ansfeats = None, None, None, None
            if self.opt.answer == 'encoder':
                if self.opt.answer_feature:
                    if self.opt.feature:
                        src, ans, feats, tgt, ansfeats, indices = batch[0]
                    else:
                        src, ans, tgt, ansfeats, indices = batch[0]
                        feats = None
                else:
                    if self.opt.feature:
                        src, ans, feats, tgt, indices = batch[0]
                    else:
                        src, ans, tgt, indices = batch[0]
                        feats = None
            else:
                if self.opt.feature:
                    src, bio, feats, tgt, indices = batch[0]
                else:
                    src, bio, tgt, indices = batch[0]
                    feats = None
            src_batch, tgt_batch = raw_batch

            #  (2) translate
            pred, predScore, predIsCopy, predCopyPosition, attn, _ = self.translator.translateBatch(src, bio, feats, tgt, ans, ansfeats)
            pred, predScore, predIsCopy, predCopyPosition, attn = list(zip(
                *sorted(zip(pred, predScore, predIsCopy, predCopyPosition, attn, indices),
                        key=lambda x: x[-1])))[:-1]

            #  (3) convert indexes to words
            predBatch = []
            for b in range(src[0].size(1)):
                n = 0
                predBatch.append(
                    self.translator.buildTargetTokens(pred[b][n], src_batch[b],
                                                 predIsCopy[b][n], predCopyPosition[b][n], attn[b][n])
                )
            # nltk BLEU evaluator needs tokenized sentences
            gold += [[r] for r in tgt_batch]
            predict += predBatch

        no_copy_mark_predict = [[word.replace('[[', '').replace(']]', '') for word in sent] for sent in predict]
        bleu = bleu_score.corpus_bleu(gold, no_copy_mark_predict)
        report_metric = bleu

        with open(ofn, 'w', encoding='utf-8') as of:
            for p in predict:
                of.write(' '.join(p) + '\n')
        return report_metric
