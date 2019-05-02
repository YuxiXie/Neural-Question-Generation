## NQG MODEL

#### codes organize

```
seq2seq_pt
├── PyBLEU
|   └── nltk_bleu_score.py
├── s2s
|   ├── modules
|   |   ├── ConcatAttention.py
|   |   ├── GlobalAttention.py
|   |   ├── Maxout.py
|   |   ├── myAdam.py
|   |   └── myRNN.py
|   ├── Beam.py
|   ├── Constants.py
|   ├── Dataset.py
|   ├── Dict.py
|   ├── Models.py
|   ├── Optim.py
|   ├── Trainslator.py
|   ├── xinit.py
|   ├── xutils.py
|   ├── Loss.py
|   ├── Train.py
|   └── Eval.py
├── CollectVocab.py
├── onlinePreprocess.py
├── trainslate.py
├── xargs.py
├── run.py
└── run_squad_qg.sh
```

#### Run.py

1. parse the command options with funtions in `xargs.py`

2. prepare the logger

3. prepare training data

    * already have both the data and vocab of tockens and indexes prepared

        - `bio.vocab.txt`; `feat.vocab.txt`

        - `vocab.txt`; `vocab.txt.20k`

        - `train.txt.bio`; `train.txt.case`; `train.txt.ner`; `train.txt.pos`

        - `train.txt.source.txt`; `train.txt.target.txt`

    * prepare_data_online in `onlinePreprocess.py`

        - dictionary of training data

    * Dataset in `s2s.Dataset.py`

        - class of dataset

        - can do batch-split and data shuffle, etc.

4. define and prepare model

    * encoder - Encoder in `s2s.Models.py`

    * decoder - Decoder in `s2s.Models.py`

    * decoder is inited by decIniter - DecInit in `s2s.Models.py`

    * generator is defined there by `nn.Sequential()`

    * the whole model - NMTModel in `s2s.Models.py`

    * translator - Translator in `s2s.Translator.py`

    * init parameters there

    * load pretrained vectors there (optional)

5. prepare development data

    * dataset - return the Dataset class of development data

    * raw - list of tockens of src and tgt data

6. define and prepare optimizer

    * use Optim in `s2s.Optim.py`

7. train

    * prepare evaluator using Evaluator in `s2s.Eval.py`

    * prepare trainer using SupervisedTrainer in `s2s.Train.py`

    * train by the `trainModel()` method of trainer

#### Train.py

1. start from `trainModel()`

    * define Criterion and use `NLLLoss` as default

    * for each epoch, use `trainEpoch()` to train the model

    * `saveModel()` for each epoch

2. in `trainEpoch()`

    * shuffle the training data batch order

    * for each batch

        - get the result of the model with the batch

            * `g_outputs`; `c_outputs`; `c_gate_values`

        - calculate loss for the result

            * update parameters and optimizer using loss

        - report when get log_interval

        - valid the model using `evalModel()` of evaluator

            * get bleu4, then report the metric recult

            * `saveModel()`

#### Eval.py

1. build for dev data and different kinds of metric

2. in `evalModel()`

    * for each dev batch (dataset, raw)

        - use translator to get the prediction

        - convert indexes into words

    * calculate bleu4 for the whole data

    * save the result (predicted sentences with copy marks)

#### Models.py

**Encoder**

* 3 x Embedding

   * word + bio + features

* rnn

   * default : GRU + bi + 512 hidden + 0.5 dropout

**Decoder**

* rnn : StackedGRU

* attention (default)

* copy mechanism (default)

```python
self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)

copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
copyProb = F.sigmoid(copyProb)
```

* maxout

```python
self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
self.maxout_pool_size = opt.maxout_pool_size

output = self.dropout(maxout)
```

* three kinds of output ( generate, copy, prob )

```python
g_outputs += [output]
c_outputs += [attn]
copyGateOutputs += [copyProb]

g_outputs = torch.stack(g_outputs)
c_outputs = torch.stack(c_outputs)
copyGateOutputs = torch.stack(copyGateOutputs)
```

**StackedGRU**

* dropout

* layer number: 1 (default)

* make it convenient when number of layers > 1

*DecInit*

* Linear layer

```python
 self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
 ```

 * tanh

 ```python
 self.tanh = nn.Tanh()

 return self.tanh(self.initer(last_enc_h))
 ```

**NMTModel**

* encoder

```python
enc_hidden, context = self.encoder(src, bio, feats)
```

* decoder_init

```python
enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)
```

* decoder

```python
g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector
      = self.decoder(tgt, enc_hidden, context, src_pad_mask, init_att)
```
