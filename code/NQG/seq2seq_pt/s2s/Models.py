import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import s2s.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from pytorch_pretrained_bert import BertModel
from torch import cuda

try:
    import ipdb
except ImportError:
    pass


class AnswerEncoder(nn.Module):
    def __init__(self, opt, vocab_size):
        self.num_directions = 2 if opt.answer_brnn else 1
        assert opt.answer_enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.answer_enc_rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(AnswerEncoder, self).__init__()

        self.word_lut = nn.Embedding(vocab_size,
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)

        self.feature = opt.answer_feature
        if self.feature:
            self.feat_lut = nn.Embedding(58, 16, padding_idx=s2s.Constants.PAD)

        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=1,
                          dropout=opt.dropout,
                          bidirectional=opt.answer_brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_answer_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_answer_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, feats, hidden=None):
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        wordEmb = self.word_lut(input[0])
        if self.feature:
            ff = feats[0]
            featsEmb = [self.feat_lut(feat) for feat in feats[0]]
            featsEmb = torch.cat(featsEmb, dim=-1)
            input_emb = torch.cat((wordEmb, featsEmb), dim=-1)
        else:
            input_emb = wordEmb


        lens = lengths
        input_emb = input_emb.transpose(0, 1)
        indices = range(len(input_emb))

        input_emb, lens, indices = zip(*sorted(zip(input_emb, lens, indices), key=lambda x: -x[1]))

        input_emb = [x.unsqueeze(0) for x in list(input_emb)]
        input_emb = torch.cat(input_emb, dim=0).transpose(0, 1)

        emb = pack(input_emb, list(lens))

        outputs, hidden_t = self.rnn(emb, hidden)

        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]

        outputs = outputs.transpose(0, 1)
        outputs, indices, h1, h2 = zip(*sorted(zip(outputs, indices, hidden_t[0], hidden_t[1]), key=lambda x: x[1]))

        outputs = [x.unsqueeze(0) for x in list(outputs)]
        outputs = torch.cat(outputs, dim=0).transpose(0, 1)

        h1 = [x.unsqueeze(0) for x in list(h1)]
        h1 = torch.cat(h1, dim=0)

        h2 = [x.unsqueeze(0) for x in list(h2)]
        h2 = torch.cat(h2, dim=0)

        hidden_t = (h1, h2)

        return hidden_t, outputs


class Encoder(nn.Module):
    def __init__(self, opt, dicts, answer_size=None):
        self.bert = opt.bert

        self.device = torch.device("cuda" if opt.gpus else "cpu")
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        
        input_size = opt.word_vec_size
        if opt.bert:
            input_size = 768

        super(Encoder, self).__init__()
        if opt.bert:
            self.word_lut = BertModel.from_pretrained('/home/xieyuxi/bert/bert-base-uncased/')
        else:
            self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=s2s.Constants.PAD)
        self.answer = opt.answer
        if self.answer == 'embedding':
            self.bio_lut = nn.Embedding(7, 16, padding_idx=s2s.Constants.PAD)  # TODO: Fix this magic number
        self.feature = opt.feature
        if self.feature:
            self.feat_lut = nn.Embedding(63, 16, padding_idx=s2s.Constants.PAD)  # TODO: Fix this magic number
        if self.answer == 'embedding':
            input_size += 16    # TODO: Fix this magic number
        if self.feature:
            input_size += 16 * 3    # TODO: Fix this magic number
        
        self.position = opt.position
        if opt.position:
            self.pos_lut = nn.Embedding(128, 16, padding_idx=s2s.Constants.PAD)  # TODO: Fix this magic number

        self.paragraph = opt.paragraph
        if self.paragraph:
            self.linear_trans = nn.Linear(opt.enc_rnn_size, opt.enc_rnn_size)
            self.update_layer = nn.Linear(2 * opt.enc_rnn_size, opt.enc_rnn_size, bias=False)
            self.gate = nn.Linear(2 * opt.enc_rnn_size, opt.enc_rnn_size, bias=False)

        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.dropout,
                          bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def gated_self_attn(self, queries, memories, mask):
        # [b, l, h] * [b, h, l] = [b, l, l]
        energies = torch.matmul(queries, memories.transpose(1, 2))
        energies = energies.masked_fill(mask.unsqueeze(1), value=-1e12)

        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        # [b, l, h*2]
        inputs = torch.cat([queries, context], dim=2)

        # [b, l, h*2] ——> [b, l, h]
        f_t =  torch.tanh(self.update_layer(inputs))
        # [b, l, h*2] ——> [b, l, h] ——> [b, 1] ?
        g_t = torch.sigmoid(self.gate(inputs))


        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def forward(self, input, bio, feats, hidden=None):
        #print(feats[0])
        lengths = input[-1].data.view(-1).tolist()  # lengths data is wrapped inside a Variable
        if self.bert:
            wordEmb = self.word_lut(input[0].transpose(0, 1), output_all_encoded_layers=False)[0]
            wordEmb = wordEmb.transpose(0, 1)
        else:
            wordEmb = self.word_lut(input[0])
        
        if self.answer == 'embedding':
            bioEmb = self.bio_lut(bio[0])
            if self.position:
                position = bio[0].transpose(0, 1)
                def get_pos(sent):
                    pos = [w for w in sent]
                    b = list(sent).index(6)
                    i = b 
                    while i < len(sent) - 1 and sent[i + 1] in [5, 6]:
                        i += 1
                    e = i
                    for i, w in enumerate(pos):
                        if i < b and sent[i] in [4, 5, 6]:
                            pos[i] = b - i 
                        elif i > e and sent[i] in [4, 5, 6]:
                            pos[i] = i - e
                        elif sent[i] in [4, 5, 6]:
                            pos[i] = 120
                    return pos
                position = [get_pos(sent) for sent in position]
                position = torch.LongTensor(position)
                position = position.transpose(0, 1)
                position = position.to(self.device)
                posEmb = self.pos_lut(position)

            if self.feature:
                featsEmb = [self.feat_lut(feat) for feat in feats[0]]
                featsEmb = torch.cat(featsEmb, dim=-1)
                input_emb = torch.cat((wordEmb, bioEmb, featsEmb), dim=-1)
            else:
                input_emb = torch.cat((wordEmb, bioEmb), dim=-1)
        else:
            if self.feature:
                ff = feats[0]
                featsEmb = [self.feat_lut(feat) for feat in feats[0]]
                featsEmb = torch.cat(featsEmb, dim=-1)
                try:
                    input_emb = torch.cat((wordEmb, featsEmb), dim=-1)
                except:
                    for f in ff:
                        print(f.size())
            else:
                input_emb = wordEmb
        emb = pack(input_emb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]

        # self attention for paragraph
        if self.paragraph:
            hid, ctxt = hidden_t

            ipt = input[0].transpose(0, 1)
            mask = (ipt == 0).byte()

            tp_outputs = outputs.transpose(0, 1)
            memories = self.linear_trans(tp_outputs)
            tp_outputs = self.gated_self_attn(tp_outputs, memories, mask)
            outputs = tp_outputs.transpose(0, 1)
        
        if self.position:
            outputs = (outputs, posEmb)

        return hidden_t, outputs


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class Decoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt

        self.bert = opt.bert

        self.layers = opt.layers
        self.input_feed = opt.input_feed

        input_size = opt.word_vec_size
        if opt.bert:
            input_size = 768

        if self.input_feed:
            input_size += opt.enc_rnn_size
            if opt.position:
                input_size += 16

        super(Decoder, self).__init__()
        if opt.bert:
            self.word_lut = BertModel.from_pretrained('/home/xieyuxi/bert/bert-base-uncased/')
        else:
            self.word_lut = nn.Embedding(dicts.size(), opt.word_vec_size, padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)

        self.is_coverage = opt.coverage

        self.position = opt.position
        if opt.position:
            self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size + 16, opt.dec_rnn_size, opt.att_vec_size, opt.coverage)  # TODO: Fix this magic number
        else:
            self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size, opt.coverage)
        
        self.dropout = nn.Dropout(opt.dropout)
        if opt.bert:
            word_vec_size = 768
        else:
            word_vec_size = opt.word_vec_size
        if opt.position:
            self.readout = nn.Linear((opt.enc_rnn_size + 16 + opt.dec_rnn_size + word_vec_size), opt.dec_rnn_size)
        else:
            self.readout = nn.Linear((opt.enc_rnn_size + opt.dec_rnn_size + word_vec_size), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.copy = opt.copy
        if self.copy:
            if opt.position:
                self.copySwitch = nn.Linear(opt.enc_rnn_size + 16 + opt.dec_rnn_size, 1)
            else:
                self.copySwitch = nn.Linear(opt.enc_rnn_size + opt.dec_rnn_size, 1)

        self.hidden_size = opt.dec_rnn_size
        self.device = torch.device("cuda" if opt.gpus else "cpu")

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, src_pad_mask, init_att):
        if self.bert:
            emb = self.word_lut(input.transpose(0, 1), output_all_encoded_layers=False)[0]
            emb = emb.transpose(0, 1)
        else:
            emb = self.word_lut(input)

        coverage_outputs = []
        g_outputs = []
        c_outputs = []
        copyGateOutputs = []
        cur_context = init_att
        self.attn.applyMask(src_pad_mask)
        precompute, coverage = None, None
        if isinstance(context, tuple):
            context = torch.cat(context, dim=-1)

        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context], 1)
            output, hidden = self.rnn(input_emb, hidden)

            if self.is_coverage:
                if coverage is None:
                    coverage = Variable(torch.zeros((context.size(1), context.size(0))))
                    coverage = coverage.to(self.device)
                cur_context, attn, precompute, next_coverage = self.attn(output, context.transpose(0, 1), precompute, coverage)
                coverage_loss = torch.sum(torch.min(attn, coverage), 1)
                coverage = next_coverage
                coverage_outputs.append(coverage_loss)
            else:
                cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            if self.copy:
                copyProb = self.copySwitch(torch.cat((output, cur_context), dim=1))
                copyProb = F.sigmoid(copyProb)
                c_outputs += [attn]
                copyGateOutputs += [copyProb]
            readout = self.readout(torch.cat((emb_t, output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
            
        g_outputs = torch.stack(g_outputs)

        if self.copy:
            c_outputs = torch.stack(c_outputs)
            copyGateOutputs = torch.stack(copyGateOutputs)
            if self.is_coverage:
                coverage_outputs = torch.stack(coverage_outputs)
                return g_outputs, c_outputs, copyGateOutputs, hidden, attn, cur_context, coverage_outputs
            else:
                return g_outputs, c_outputs, copyGateOutputs, hidden, attn, cur_context
        else:
            if self.is_coverage:
                coverage_outputs = torch.stack(coverage_outputs)
                return g_outputs, hidden, attn, cur_context, coverage_outputs
            else:
                return g_outputs, hidden, attn, cur_context


class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        if opt.answer == 'encoder':
            self.answer_enc_rnn_size = opt.answer_enc_rnn_size
            self.initer = nn.Linear(self.answer_enc_rnn_size, self.dec_rnn_size)
        else:
            self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):
        # batchSize = last_enc_h.size(0)
        # dim = last_enc_h.size(1)
        return self.tanh(self.initer(last_enc_h))


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, decIniter, answer_encoder=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.answer_encoder = answer_encoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        if isinstance(context, tuple):
            batch_size = context[0].size(1)
            h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions + 16)
            context = torch.cat(context, dim=-1)
        else:
            batch_size = context.size(1)
            h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()

    def forward(self, input):
        """
        (wrap(srcBatch), lengths), \
               (wrap(bioBatch), lengths), ((wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
        (wrap(srcBatch), lengths), \
               (wrap(ansBatch), ans_lengths), ((wrap(x) for x in featBatches), lengths), \
               (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch)), \
               indices
        """
        # ipdb.set_trace()
        src = input[0]
        feats = None
        if self.encoder.feature:
            tgt = input[3][0][:-1]  # exclude last target from inputs
            feats = input[2]
        else:
            tgt = input[2][0][:-1]
        src_pad_mask = Variable(src[0].data.eq(s2s.Constants.PAD).transpose(0, 1).float(),
                                requires_grad=False, volatile=False)

        bio, ans, ans_feats = None, None, None
        if self.encoder.answer == 'encoder':
            ans = input[1]
            if self.answer_encoder.feature:
                if self.encoder.feature:
                    ans_feats = input[4]
                else:
                    ans_feats = input[3]
        elif self.encoder.answer == 'embedding':
            bio = input[1]

        #print(feats[0])
        enc_hidden, context = self.encoder(src, bio, feats)

        init_att = self.make_init_att(context)
        if self.encoder.answer == 'encoder':
            ans_hidden, _ = self.answer_encoder(ans, ans_feats)
            ans_hidden = torch.cat((ans_hidden[0], ans_hidden[1]), dim=-1)
            ans_hidden = self.decIniter(ans_hidden).unsqueeze(0)
            enc_hidden = ans_hidden
        else:
            enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hidden

        if self.decoder.copy:
            if self.decoder.is_coverage:
                g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector, coverage_out \
                    = self.decoder(tgt, enc_hidden, context,  src_pad_mask, init_att)
                return g_out, c_out, c_gate_out, coverage_out
            else:
                g_out, c_out, c_gate_out, dec_hidden, _attn, _attention_vector \
                    = self.decoder(tgt, enc_hidden, context,  src_pad_mask, init_att)
                return g_out, c_out, c_gate_out
        else:
            if self.decoder.is_coverage:
                g_out, dec_hidden, _attn, _attention_vector, coverage_out \
                    = self.decoder(tgt, enc_hidden, context, src_pad_mask, init_att)
                return g_out, coverage_out
            else:
                g_out, dec_hidden, _attn, _attention_vector \
                    = self.decoder(tgt, enc_hidden, context, src_pad_mask, init_att)
                return g_out
