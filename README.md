# NQG

* This repository builds a seq2seq-based model which can test different techniques of the state-of-arts in the field of neural question generation.

* The raw code comes from the implementation code for the paper "[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/abs/1704.01792)"

```
@article{zhou2017neural,
  title={Neural Question Generation from Text: A Preliminary Study},
  author={Zhou, Qingyu and Yang, Nan and Wei, Furu and Tan, Chuanqi and Bao, Hangbo and Zhou, Ming},
  journal={arXiv preprint arXiv:1704.01792},
  year={2017}
}
```

## About this code

Experiments contain the following techniques:

* answer encoding

    - bio embedding in source

    - separate answer encoding
   
* feature encoding

    - POS tag

    - NER tag

    - case tag

* copy mechanism

* question word mechanism

* BERT

* gated self-attention for paragraph-level encoding

---

## How to run

### Prepare the dataset and code

Make an experiment home folder for NQG data - ```squad```:
```bash
mkdir -p $NQG_HOME/data
cd $NQG_HOME/data
wget https://res.qyzhou.me/redistribute.zip
unzip redistribute.zip
```
Put the data in the folder `$NQG_HOME/code/data/giga` and organize them as:
```
nqg
├── code
│   └── NQG
│       └── seq2seq_pt
└── data
    └── redistribute
        ├── QG
        │   ├── dev
        │   ├── test
        │   ├── test_sample
        │   └── train
        └── raw
```
Then collect vocabularies:
```bash
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.source.txt \
       $NQG_HOME/data/redistribute/QG/train/train.txt.target.txt \
       $NQG_HOME/data/redistribute/QG/train/vocab.txt
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.bio \
       $NQG_HOME/data/redistribute/QG/train/bio.vocab.txt
python $NQG_HOME/code/NQG/seq2seq_pt/CollectVocab.py \
       $NQG_HOME/data/redistribute/QG/train/train.txt.pos \
       $NQG_HOME/data/redistribute/QG/train/train.txt.ner \
       $NQG_HOME/data/redistribute/QG/train/train.txt.case \
       $NQG_HOME/data/redistribute/QG/train/feat.vocab.txt
head -n 20000 $NQG_HOME/data/redistribute/QG/train/vocab.txt > $NQG_HOME/data/redistribute/QG/train/vocab.txt.20k
```

### Setup the environment
#### Package Requirements:
```
nltk scipy numpy pytorch
```
**PyTorch version**: This code requires PyTorch v0.4.0.

**Python version**: This code requires Python3.

**Warning**: Older versions of NLTK have a bug in the PorterStemmer. Therefore, a fresh installation or update of NLTK is recommended.

#### Without Docker
```bash
bash $NQG_HOME/code/NQG/seq2seq_pt/run_squad_qg.sh $NQG_HOME/data/redistribute/QG $NQG_HOME/code/NQG/seq2seq_pt
```
#### Inside the docker:
```bash
bash code/NQG/seq2seq_pt/run_squad_qg.sh /workspace/data/redistribute/QG /workspace/code/NQG/seq2seq_pt
```
