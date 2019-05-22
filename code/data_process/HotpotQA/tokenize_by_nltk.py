import json
import codecs
import nltk
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk


json_load = lambda x : json.load(codecs.open(x, 'r', encoding='utf-8'))

def data_dump(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data) + '\n')

def process(data, mode, ref=None):
    SRCs, TGTs, ANSs, POSs, NERs, CASEs = [], [], [], [], [], []
    for sample in tqdm(data):
        flag = True

        qu = sample['question']
        qu = sent_tokenize(qu.strip())
        qu = [word_tokenize(sent.strip()) for sent in qu]
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
            ans = [word_tokenize(sent.strip()) for sent in ans]

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

            passage = [[word_tokenize(sent.strip()) for sent in c] for c in context]
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
                                cs.append('LOW')
                            else:
                                cs.append('UP')
                        cs = ' '.join(cs)
                        cse.append(cs)
                    cse = ' <sep> '.join(cse)
                    case.append(cse)
                case = ' <para-sep> '.join(case)

                pos = [[pos_tag(sent) for sent in c] for c in passage]

                ner = [[ne_chunk(sent) for sent in c] for c in pos]
                ner_tags = []
                for c in ner:
                    ner_tag = []
                    for sent in c:
                        ners = []
                        for w in sent:
                            if isinstance(w, nltk.tree.Tree):
                                for i in range(len(w)):
                                    ners.append(w.label())
                            else:
                                ners.append('O')
                        ners = ' '.join(ners)
                        ner_tag.append(ners)
                    ner_tag = ' <sep> '.join(ner_tag)
                    ner_tags.append(ner_tag)
                ner = ' <para-sep> '.join(ner_tags)

                pos = [[[w[1] for w in sent] for sent in c] for c in pos]
                pos = [[' '.join(sent) for sent in c] for c in pos]
                pos = [' <sep> '.join(c) for c in pos]
                pos = ' <para-sep> '.join(pos)

                SRCs.append(psg)
                ANSs.append(ans)
                POSs.append(pos)
                NERs.append(ner)
                CASEs.append(case)
    
    return SRCs, TGTs, ANSs, POSs, NERs, CASEs


if __name__ == '__main__':
    raw = json_load('raw/hotpot_train_v1.1.json')
    src, tgt, ans, pos, ner, case = process(raw, 'train')
    data_dump(src, 'train-nltk/train.src.txt')
    data_dump(tgt, 'train-nltk/train.tgt.txt')
    data_dump(ans, 'train-nltk/train.ans.txt')
    data_dump(pos, 'train-nltk/train.pos.txt')
    data_dump(ner, 'train-nltk/train.ner.txt')
    data_dump(case, 'train-nltk/train.case.txt')

    train = src 
    raw = json_load('raw/hotpot_dev_v1.1.json')
    src, tgt, ans, pos, ner, case = process(raw, 'dev', train)
    data_dump(src, 'dev-nltk/dev.src.txt')
    data_dump(tgt, 'dev-nltk/dev.tgt.txt')
    data_dump(ans, 'dev-nltk/dev.ans.txt')
    data_dump(pos, 'dev-nltk/dev.pos.txt')
    data_dump(ner, 'dev-nltk/dev.ner.txt')
    data_dump(case, 'dev-nltk/dev.case.txt')
    