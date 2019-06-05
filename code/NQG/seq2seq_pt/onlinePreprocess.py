import logging
import torch
import s2s
from pytorch_pretrained_bert import BertTokenizer

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 512
report_every = 100000
shuffle = 1

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = s2s.Dict([s2s.Constants.PAD_WORD, s2s.Constants.UNK_WORD,
                      s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = s2s.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(copy, answer, feature, answer_feature, srcFile, bioFile, 
             featFiles, tgtFile, srcDicts, bioDicts, featDicts, tgtDicts, 
             ansFile=None, ansDicts=None, ansfeatFiles=None, ansfeatDicts=None,
             bert=False, tokenizer=None):
    src, tgt, ans = [], [], []
    bio = []
    ansfeats = []
    feats = []
    switch, c_tgt = [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    if answer == 'encoder':
        ansF = open(ansFile, encoding='utf-8')
        if answer_feature:
            ansfeatFs = [open(x, encoding='utf-8') for x in ansfeatFiles]
    elif answer == 'embedding':
        bioF = open(bioFile, encoding='utf-8')
    if feature:
        featFs = [open(x, encoding='utf-8') for x in featFiles]

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()
        if answer == 'encoder':
            aline = ansF.readline()
            if answer_feature:
                aflines = [x.readline() for x in ansfeatFs]
        elif answer == 'embedding':
            bioLine = bioF.readline()
        if feature:
            featLines = [x.readline() for x in featFs]

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "" or (answer == 'encoder' and aline == ""):
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()
        if answer == 'encoder':
            aline = aline.strip()
            if answer_feature:
                aflines = [line.strip() for line in aflines]
        elif answer == 'embedding':
            bioLine = bioLine.strip()
        if feature:
            featLines = [line.strip() for line in featLines]

        # source and/or target are empty
        if sline == "" or tline == "" or (answer == 'encoder' and aline == ""):
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        tgtWords = tline.split(' ')
        if answer == 'encoder':
            ansWords = aline.split(' ')
            if answer_feature:
                afWords = [x.split(' ') for x in aflines]
        elif answer == 'embedding':
            bioWords = bioLine.split(' ')
        if feature:
            featWords = [x.split(' ') for x in featLines]
        
        if bert:
            max_seq_length = seq_length + 2
        else:
            max_seq_length = seq_length

        if len(srcWords) <= seq_length and len(tgtWords) <= max_seq_length:
            if bert:
                src += [torch.LongTensor(tokenizer.convert_tokens_to_ids(srcWords))]
            else:
                src += [srcDicts.convertToIdx(srcWords, s2s.Constants.UNK_WORD)]
            if answer == 'encoder':
                ans += [ansDicts.convertToIdx(ansWords, s2s.Constants.UNK_WORD)]
                if answer_feature:
                    ansfeats += [[ansfeatDicts.convertToIdx(x, s2s.Constants.UNK_WORD) for x in afWords]]
            elif answer == 'embedding':
                bio += [bioDicts.convertToIdx(bioWords, s2s.Constants.UNK_WORD)]
            if feature:
                feats += [[featDicts.convertToIdx(x, s2s.Constants.UNK_WORD) for x in featWords]]
            if bert:
                tgt += [torch.LongTensor(tokenizer.convert_tokens_to_ids(tgtWords))]
            else:
                tgt += [tgtDicts.convertToIdx(tgtWords, s2s.Constants.UNK_WORD,
                                              s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD)]
            if copy:
                if bert:
                    switch_buf = [0] * len(tgtWords)
                    c_tgt_buf = [0] * len(tgtWords)
                else:
                    switch_buf = [0] * (len(tgtWords) + 2)
                    c_tgt_buf = [0] * (len(tgtWords) + 2)
                for idx, tgt_word in enumerate(tgtWords):
                    if bert:
                        word_id = tokenizer.convert_tokens_to_ids([tgt_word])
                        word_id = word_id[0]
                    else:
                        word_id = tgtDicts.lookup(tgt_word, None)
                    if word_id is None:
                        if tgt_word in srcWords:
                            copy_position = srcWords.index(tgt_word)
                            if bert:
                                switch_buf[idx] = 1
                                c_tgt_buf[idx] = copy_position
                            else:
                                switch_buf[idx + 1] = 1
                                c_tgt_buf[idx + 1] = copy_position
                switch.append(torch.FloatTensor(switch_buf))
                c_tgt.append(torch.LongTensor(c_tgt_buf))

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    if answer == 'encoder':
        ansF.close()
        if answer_feature:
            for x in ansfeatFs:
                x.close()
    elif answer == 'embedding':
        bioF.close()
    if feature:
        for x in featFs:
            x.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        if answer == 'encoder':
            ans = [ans[idx] for idx in perm]
            if answer_feature:
                ansfeats = [ansfeats[idx] for idx in perm]
        elif answer == 'embedding':
            bio = [bio[idx] for idx in perm]
        if feature:
            feats = [feats[idx] for idx in perm]
        if copy:
            switch = [switch[idx] for idx in perm]
            c_tgt = [c_tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    if answer == 'encoder':
        ans = [ans[idx] for idx in perm]
        if answer_feature:
            ansfeats = [ansfeats[idx] for idx in perm]
        else:
            ansfeats = None
    elif answer == 'embedding':
        bio = [bio[idx] for idx in perm]
    else:
        bio = None
        ans, ansfeats = None, None
    if feature:
        feats = [feats[idx] for idx in perm]
    else:
        feats = None
    if copy:
        switch = [switch[idx] for idx in perm]
        c_tgt = [c_tgt[idx] for idx in perm]
    else:
        switch, c_tgt = None, None

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, bio, feats, tgt, switch, c_tgt, ans, ansfeats


def prepare_data_online(copy, answer, feature, ans_feature, 
                        train_src, src_vocab, train_bio, bio_vocab,
                        train_feats, feat_vocab, train_tgt, tgt_vocab,
                        train_ans=None, ans_vocab=None, train_ans_feats=None,
                        bert=False):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)
    dicts['bio'], dicts['feat'], dicts['ans'], dicts['ans-feat'] = None, None, None, None
    if answer == 'embedding':
        dicts['bio'] = initVocabulary('bio', [train_bio], bio_vocab, 0)
    if feature:
        dicts['feat'] = initVocabulary('feat', [train_feats], feat_vocab, 0)

    if answer == 'encoder':
        dicts['ans'] = initVocabulary('answer', [train_ans], ans_vocab, 0)
        if ans_feature:
            dicts['ans-feat'] = initVocabulary('ans-feat', [train_ans_feats], feat_vocab, 0)

    tokenizer = None
    if bert:
        tokenizer = BertTokenizer.from_pretrained('/home/xieyuxi/bert/bert-base-uncased-vocab.txt')
    
    logger.info('Preparing training ...')
    train = {}
    train['src'], train['bio'], train['feats'], \
    train['tgt'], train['switch'], train['c_tgt'], \
    train['ans'], train['ans-feats'] = makeData(copy, answer, feature, ans_feature, train_src,
                                                train_bio, train_feats, train_tgt,
                                                dicts['src'], dicts['bio'],
                                                dicts['feat'], dicts['tgt'],
                                                train_ans, dicts['ans'],
                                                train_ans_feats, dicts['ans-feat'], 
                                                bert, tokenizer)

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
              }
    return dataset
