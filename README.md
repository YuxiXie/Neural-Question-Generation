# NQG Package (A Preliminary Version -- the updated one will be released in a few months)

* also refer to (Package for Neural Question Generation from WING-NUS)[https://github.com/YuxiXie/OpenNQG] for a revised version

* This repository builds a seq2seq-based model which can test different techniques of the state-of-arts in the field of neural question generation.

* The based code comes from the implementation code for the paper "[Neural Question Generation from Text: A Preliminary Study](https://arxiv.org/abs/1704.01792)"

```
@article{zhou2017neural,
  title={Neural Question Generation from Text: A Preliminary Study},
  author={Zhou, Qingyu and Yang, Nan and Wei, Furu and Tan, Chuanqi and Bao, Hangbo and Zhou, Ming},
  journal={arXiv preprint arXiv:1704.01792},
  year={2017}
}
```

---

## About this code

### Experiments on model mechanisms contain :

**include answer info or not**
1. bio embedding in source
2. separate answer encoder

**position embedding used in cross attention**

**question word**
```
@inproceedings{sun-etal-2018-answer,
    title = "Answer-focused and Position-aware Neural Question Generation",
    author = "Sun, Xingwu and Liu, Jing  and Lyu, Yajuan  and He, Wei  and Ma, Yanjun  and Wang, Shi",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1427",
    pages = "3930--3939"
}
```
   
**feature encoding**
1. Part-of-Speech tagger
2. (7-class) Named-Entity-Recognition tagger
3. Case tagger
```
@inproceedings{Manning2014TheSC,
  title={The Stanford CoreNLP Natural Language Processing Toolkit},
  author={Christopher D. Manning and Mihai Surdeanu and John Bauer and Jenny Rose Finkel and Steven Bethard and David McClosky},
  booktitle={ACL},
  year={2014}
}
```

**copy mechanism**
```
@article{Gu2016IncorporatingCM,
  title={Incorporating Copying Mechanism in Sequence-to-Sequence Learning},
  author={Jiatao Gu and Zhengdong Lu and Hang Li and Victor O. K. Li},
  journal={CoRR},
  year={2016},
  volume={abs/1603.06393}
}
```

**coverage mechanism**
```
@inproceedings{See2017GetTT,
  title={Get To The Point: Summarization with Pointer-Generator Networks},
  author={Abigail See and Peter J. Liu and Christopher D. Manning},
  booktitle={ACL},
  year={2017}
}
```

**BERT**
_PS: Implementing way in this version has been discarded and replaced by a better usage in the updated one_
```
@article{Devlin2018BERTPO,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova},
  journal={CoRR},
  year={2018},
  volume={abs/1810.04805}
}
```

**paragraph-level encoding**
  use gated self-attention
```
@inproceedings{zhao-etal-2018-paragraph,
    title = {Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks},
    author = {Zhao, Yao and Ni, Xiaochuan and Ding, Yuanyuan and Ke, Qifa},
    booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
    year = {2018}
}
```

### Experiments on datasets contain :

**SQuAD**
```
@inproceedings{Rajpurkar2016SQuAD10,
  title={SQuAD: 100, 000+ Questions for Machine Comprehension of Text},
  author={Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy S. Liang},
  booktitle={EMNLP},
  year={2016}
}
```

**HotpotQA**
```
@inproceedings{Yang2018HotpotQAAD,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Zhilin Yang and Peng Qi and Saizheng Zhang and Yoshua Bengio and William W. Cohen and Ruslan R. Salakhutdinov and Christopher D. Manning},
  booktitle={EMNLP},
  year={2018}
}
```

---

## How to run

### Prepare the dataset and code

Make an experiment home folder for NQG data - **squad**:
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
bash $NQG_HOME/code/NQG/seq2seq_pt/run_qg.sh $NQG_HOME/data/redistribute/QG $NQG_HOME/code/NQG/seq2seq_pt
```
#### Inside the docker:
```bash
bash code/NQG/seq2seq_pt/run_qg.sh /workspace/data/redistribute/QG /workspace/code/NQG/seq2seq_pt
```
