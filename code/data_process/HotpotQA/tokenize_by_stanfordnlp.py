import re
import json
import codecs
from tqdm import tqdm
from nltk import sent_tokenize
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag import StanfordPOSTagger, StanfordNERTagger


json_load = lambda x : json.load(codecs.open(x, 'r', encoding='utf-8'))

def data_dump(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data) + '\n')

def filter_questions(src, tgt, ans, case, pos, ner):
    paras, answers, qus, cases = [], [], [], []
    for index, qu in enumerate(tgt):
        if qu not in qus:
            qus.append(qu)
            paras.append(src[index])
            answers.append(ans[index])
            cases.append(case[index])
    return paras, qus, answers, cases

def process(data, tokenizer, postagger, nertagger, mode, ref=None):
    SRCs, TGTs, ANSs, POSs, NERs, CASEs = [], [], [], [], [], []
    
    for sample in tqdm(data):
        flag = True

        qu = sample['question']
        qu = sent_tokenize(qu.strip())
        qu = [tokenizer.tokenize(sent.strip()) for sent in qu]
        qu = [' '.join(sent) for sent in qu]
        qu = ' <sep> '.join(qu)
        qu = qu.lower()

        if mode == 'train':
            if qu not in TGTs:
                TGTs.append(qu)
            else:
                flag = False
        
        if flag:

            ans = sample['answer']
            ans = sent_tokenize(ans.strip())
            ans = [tokenizer.tokenize(sent.strip()) for sent in ans]

            ans = [' '.join(sent) for sent in ans]
            ans = ' <sep> '.join(ans)
            ans = ans.lower()

            index = []
            for c in sample['supporting_facts']:
                if c[1] not in index:
                    index.append(c[1])
            context = []
            for i in index:
                if i < len(sample['context']):
                    context.append(sample['context'][i][1])

            passage = [[tokenizer.tokenize(sent.strip()) for sent in c] for c in context]
            psg = [[' '.join(sent) for sent in c] for c in passage]
            psg = [' <sep> '.join(c) for c in psg]
            psg = ' <para-sep> '.join(psg)
            psg = psg.lower()

            if mode == 'dev' and psg in ref:
                flag = False

            if flag:
                case = []
                for c in passage:
                    cse = []
                    for sent in c:
                        cs = []
                        for w in sent:
                            if w.islower():
                                if re.match(r'[\.?!\'",;:]+', w):
                                    cs.append(w)
                                else:
                                    cs.append('LOW')
                            else:
                                cs.append('UP')
                        cs = ' '.join(cs)
                        cse.append(cs)
                    cse = ' <sep> '.join(cse)
                    case.append(cse)
                case = ' <para-sep> '.join(case)

                pos = [[postagger.tag(sent) for sent in c] for c in passage]
                pos = [[[w[1] for w in sent] for sent in c] for c in pos]
                pos = [[' '.join(sent) for sent in c] for c in pos]
                pos = [' <sep> '.join(c) for c in pos]
                pos = ' <para-sep> '.join(pos)

                ner = [[nertagger.tag(sent) for sent in c] for c in passage]
                ner = [[[w[1] for w in sent] for sent in c] for c in ner]
                ner = [[' '.join(sent) for sent in c] for c in ner]
                ner = [' <sep> '.join(c) for c in ner]
                ner = ' <para-sep> '.join(ner)   
                
                SRCs.append(psg)
                ANSs.append(ans)
                POSs.append(pos)
                NERs.append(ner)
                CASEs.append(case)
                if mode == 'dev':
                    TGTs.append(qu)
    
    return SRCs, TGTs, ANSs, POSs, NERs, CASEs

if __name__ == '__main__':
    data = json_load('raw/hotpot_train_v1.1.json')
    
    tokenizer = StanfordTokenizer(path_to_jar = '/home/xieyuxi/stanfordnlp/postagger/stanford-postagger.jar')
    postagger = StanfordPOSTagger(model_filename='/home/xieyuxi/stanfordnlp/postagger/models/english-bidirectional-distsim.tagger', \
                                 path_to_jar = '/home/xieyuxi/stanfordnlp/postagger/stanford-postagger.jar')
    nertagger = StanfordNERTagger(model_filename='/home/xieyuxi/stanfordnlp/ner/classifiers/english.muc.7class.distsim.crf.ser.gz', \
                                   path_to_jar = '/home/xieyuxi/stanfordnlp/ner/stanford-ner.jar')

    src, tgt, ans, pos, ner, case = process(data, tokenizer, postagger, nertagger, 'train')

    data_dump(src, 'train-stanfordnlp/train.src.txt')
    data_dump(tgt, 'train-stanfordnlp/train.tgt.txt')
    data_dump(ans, 'train-stanfordnlp/train.ans.txt')
    data_dump(pos, 'train-stanfordnlp/train.pos.txt')
    data_dump(ner, 'train-stanfordnlp/train.ner.txt')
    data_dump(case, 'train-stanfordnlp/train.case.txt')

    train = src
    data = json_load('raw/hotpot_dev_v1.1.json')
    src, tgt, ans, pos, ner, case = process(data, tokenizer, postagger, nertagger, 'dev', train)

    data_dump(src, 'dev-stanfordnlp/dev.src.txt')
    data_dump(tgt, 'dev-stanfordnlp/dev.tgt.txt')
    data_dump(ans, 'dev-stanfordnlp/dev.ans.txt')
    data_dump(pos, 'dev-stanfordnlp/dev.pos.txt')
    data_dump(ner, 'dev-stanfordnlp/dev.ner.txt')
    data_dump(case, 'dev-stanfordnlp/dev.case.txt')
