from __future__ import division

import os
import s2s
import torch
import torch.nn as nn
from torch import cuda
from s2s.Eval import Evaluator
import math
import time
import logging

try:
    import ipdb
except ImportError:
    pass

from nltk.translate import bleu_score


class SupervisedTrainer(object):

    def __init__(self, model, loss, evaluator, optim, logger, opt, trainData, validData, dataset):
        self.model = model
        self.loss = loss
        self.translator = evaluator.translator
        self.optim = optim
        self.logger = logger
        self.opt = opt
        self.trainData = trainData
        self.validData = validData
        self.dataset = dataset
        self.evaluator = evaluator
        self.totalBatchCount = 0

    def saveModel(self, epoch, metric=None):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = self.model.generator.module.state_dict() if len(
            self.opt.gpus) > 1 else self.model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': self.dataset['dicts'],
            'opt': self.opt,
            'epoch': epoch,
            'optim': self.optim
        }
        save_model_path = 'model'
        if self.opt.save_path:
            if not os.path.exists(self.opt.save_path):
                os.makedirs(self.opt.save_path)
            save_model_path = self.opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_dev_metric_{1}_e{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(self, epoch, train_dataset, devData):
        if self.opt.extra_shuffle and epoch > self.opt.curriculum:
            self.logger.info('Shuffling...')
            self.trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(self.trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        report_bleu, report_batch = 0, 0

        start = time.time()

        for i in range(len(self.trainData)):
            self.totalBatchCount += 1
            """
            (wrap(srcBatch), lengths), \
               (wrap(bioBatch), lengths), ((wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
            """
            batchIdx = batchOrder[i] if epoch > self.opt.curriculum else i
            batch = self.trainData[batchIdx][:-1]  # exclude original indices

            self.model.zero_grad()

            if self.opt.feature:
                index = 3
            else:
                index = 2

            if self.loss.copy:
                g_outputs, c_outputs, c_gate_values = self.model(batch)
                copy_switch = batch[index][1][1:]
                c_targets = batch[index][2][1:]
                targets = batch[index][0][1:]  # exclude <s> from targets
                loss, res_loss, num_correct = self.loss.loss_function(g_outputs, targets,
                    self.model.generator, c_outputs, copy_switch, c_targets, c_gate_values)
            else:
                g_outputs = self.model(batch)
                targets = batch[index][0][1:]  # exclude <s> from targets
                loss, res_loss, num_correct = self.loss.loss_function(g_outputs, targets, self.model.generator)

            if math.isnan(res_loss) or res_loss > 1e20:
                self.logger.info('catch NaN')
                ipdb.set_trace()

            # update the parameters
            loss.backward()
            self.optim.step()

            if self.opt.result_path:
                '''src, bio, feats, tgt, indices, src_batch, tgt_batch'''
                processed_train, raw_train = train_dataset[0][batchIdx], train_dataset[1][batchIdx]
                each_bleu = self.evaluator.get_bleu(processed_train, raw_train)
                report_bleu += each_bleu
                report_batch += 1

            num_words = targets.data.ne(s2s.Constants.PAD).sum().item()
            report_loss += res_loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_num_correct += num_correct
            total_words += num_words
            if i % self.opt.log_interval == -1 % self.opt.log_interval:
                ppl = math.exp(min((report_loss / report_tgt_words), 16))

                self.logger.info(
                    "Epoch %2d, %6d/%5d/%5d; acc: %6.2f; loss: %6.2f; words: %5d; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                    (epoch, self.totalBatchCount, i + 1, len(self.trainData),
                     report_num_correct / report_tgt_words * 100,
                     report_loss,
                     report_tgt_words,
                     ppl,
                     report_src_words / max((time.time() - start), 1.0),
                     report_tgt_words / max((time.time() - start), 1.0),
                     time.time() - start))

                if self.opt.result_path:
                    if i % self.opt.eval_per_batch == -1 % self.opt.eval_per_batch:
                        bleu = report_bleu / report_batch
                        if not os.path.exists(self.opt.result_path):
                            os.makedirs(self.opt.result_path)
                        save_perform_path = self.opt.result_path + os.path.sep + 'train.txt'
                        with open(save_perform_path, 'a', encoding='utf-8') as fout:
                            fout.write(str(epoch) + '\t' + str(self.totalBatchCount) + '\t' + str(ppl) + '\t' + str(bleu) + '\n')
                        report_bleu = report_batch = 0
                    else:
                        if not os.path.exists(self.opt.result_path):
                            os.makedirs(self.opt.result_path)
                        save_perform_path = self.opt.result_path + os.path.sep + 'train.txt'
                        with open(save_perform_path, 'a', encoding='utf-8') as fout:
                            fout.write(str(epoch) + '\t' + str(self.totalBatchCount) + '\t' + str(ppl) + '\n')

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()
            
            if self.validData is not None and self.totalBatchCount % self.opt.eval_per_batch == -1 % self.opt.eval_per_batch \
                    and self.totalBatchCount >= self.opt.start_eval_batch:
                self.model.eval()
                self.logger.warning("Set model to {0} mode".format('train' if self.model.decoder.dropout.training else 'eval'))
                valid_bleu = self.evaluator.evalModel(self.model, self.validData)
                self.model.train()
                self.logger.warning("Set model to {0} mode".format('train' if self.model.decoder.dropout.training else 'eval'))
                self.model.decoder.attn.mask = None
                self.logger.info('Validation Score: %g' % (valid_bleu * 100))
                if valid_bleu >= self.optim.best_metric:
                    self.saveModel(epoch, valid_bleu)
                self.optim.updateLearningRate(valid_bleu, epoch)

                if self.opt.result_path:
                    ### calculate loss ###
                    report_loss, report_tgt_words = 0, 0
                    self.model.eval()
                    self.logger.warning("Set model to {0} mode".format('train' if self.model.decoder.dropout.training else 'eval'))
                    for i in range(len(devData)):
                        batch = devData[i][:-1]
                        if self.loss.copy:
                            g_outputs, c_outputs, c_gate_values = self.model(batch)
                            targets = batch[index][0][1:]  # exclude <s> from targets
                            copy_switch = batch[index][1][1:]
                            c_targets = batch[index][2][1:]
                            loss, res_loss, num_correct = self.loss.loss_function(g_outputs, targets,
                                self.model.generator, c_outputs, copy_switch, c_targets, c_gate_values)
                        else:
                            g_outputs = self.model(batch)
                            targets = batch[index][0][1:]  # exclude <s> from targets
                            loss, res_loss, num_correct = self.loss.loss_function(g_outputs, targets, self.model.generator)
                        num_words = targets.data.ne(s2s.Constants.PAD).sum().item()
                        report_loss += res_loss
                        report_tgt_words += num_words
                    valid_ppl = math.exp(min((report_loss / report_tgt_words), 16))
                    self.model.train()
                    self.logger.warning("Set model to {0} mode".format('train' if self.model.decoder.dropout.training else 'eval'))
                    ### save results ###
                    if not os.path.exists(self.opt.result_path):
                        os.makedirs(self.opt.result_path)
                    save_perform_path = self.opt.result_path + os.path.sep + 'dev.txt'
                    with open(save_perform_path, 'a', encoding='utf-8') as fout:
                        fout.write(str(epoch) + '\t' + str(self.totalBatchCount) + '\t' + str(valid_ppl) + '\t' + str(valid_bleu) + '\n')

        return total_loss / total_words, total_num_correct / total_words

    def trainModel(self, train_dataset, devData):
        self.logger.info(self.model)
        self.model.train()
        self.logger.warning("Set model to {0} mode".format('train' if self.model.decoder.dropout.training else 'eval'))

        start_time = time.time()

        for epoch in range(self.opt.start_epoch, self.opt.epochs + 1):
            self.logger.info('')
            #  (1) train for one epoch on the training set
            train_loss, train_acc = self.trainEpoch(epoch, train_dataset, devData)
            train_ppl = math.exp(min(train_loss, 100))
            self.logger.info('Train perplexity: %g' % train_ppl)
            self.logger.info('Train accuracy: %g' % (train_acc * 100))
            self.logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
            self.saveModel(epoch)
