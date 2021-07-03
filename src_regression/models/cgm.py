import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_size):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, query=None, mask=None):
        '''

        :param query:   (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''
        if query is None:
            h = torch.tanh(self.w_1(memory))  # node, hidden
        else:
            h = torch.tanh(self.w_1(memory) + self.w_2(query))
        score = torch.squeeze(self.u(h), -1)  # node,
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node,
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2)
        return s


class RGCN(nn.Module):
    def __init__(self, in_features, out_features, relation_num):
        super(RGCN, self).__init__()
        self.relation_num = relation_num
        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)
        self.linears = [self.linear_1, self.linear_2]
        self.activation = nn.Tanh()

    def gcn(self, relation, input, adj):
        support = self.linears[relation](input)
        output = torch.sparse.mm(adj, support)
        return output

    def forward(self, input, adjs):
        '''

        :param input:   (node, hidden)
        :param adjs:    (node, node)
        :return:
        '''
        transform = []
        for r in range(self.relation_num):
            transform.append(self.gcn(r, input, adjs[r]))
        # (node, relation, hidden) -> (node, hidden)
        return self.activation(torch.sum(torch.stack(transform, 1), 1))


class SLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, relation_num, dropout):
        super(SLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.Wh = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.Wt = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 5, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 5)
        self.rgcn = RGCN(hidden_size, hidden_size, relation_num)

    def forward(self, x, h, c, g, h_t, adjs):
        '''

        :param x:   (node, emb)
            embedding of the node, news and initial node embedding
        :param h:   (node, hidden)
            hidden state from last layer
        :param c:   candidate from last layer
        :param g:   (hidden)
            hidden state of the global node
        :param h_t:   (node, hidden)
            hidden state from last time
        :param adj:   (node, node)
            if use RGCN, there should be multiple gcns, each one for a relation
        :return:
        '''
        # adjs = [adj]
        hn = self.rgcn(h, adjs)
        gates = self.Wh(self.dropout(h)) + self.U(self.dropout(x)) + self.Wn(self.dropout(hn)) + self.Wt(self.dropout(h_t)) + torch.unsqueeze(self.V(g), 0)
        i, f, o, u, t = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(t) * h_t
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c


class GLSTMCell(nn.Module):
    def __init__(self, hidden_size, attn_pooling, dropout):
        super(GLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.W = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.T = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.t = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 5)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.attn_pooling = attn_pooling

    def forward(self, g, c_g, t_g, t_c, h, c, node_emb):
        '''

        :param g:   global hidden from last layer
        :param c_g: global candidate from last layer
        :param t_g: global hidden from last time
        :param h:   (node, hidden)
            hidden states of all nodes
        :param c:   candidates of all nodes
        :return:
        '''
        h_avg = self.attn_pooling(h, node_emb)
        # the gates are calculated according to h
        gates = self.W(self.dropout(g)) + self.U(self.dropout(h_avg)) + self.T(self.dropout(t_g))
        i, f, o, u, t = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c_g + torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(t) * t_c
        '''
        f, o, t = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg) + self.T(t_g)), self.hidden_size, dim=-1)
        i_w = torch.sigmoid(torch.unsqueeze(self.w(g), 0) + self.u(h))  # node, hidden
        i_w = F.softmax(i_w, 0)
        # c is calculated according to c
        new_c = f * c_g + torch.sum(c * i_w, 0) + t * t_c
        '''
        new_g = o * torch.tanh(new_c)
        return new_g, new_c

    def init_forward(self, t_g, t_c, h, c, node_emb):
        h_avg = self.attn_pooling(h, node_emb)
        # the gates are calculated according to h
        gates = self.dropout(self.W(t_g) + self.U(h_avg))
        i, f, o, u, _ = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * t_c + torch.sigmoid(i) * torch.tanh(u)
        '''
        f, o, _ = torch.split(torch.sigmoid(self.W(t_g) + self.U(h_avg)), self.hidden_size, dim=-1)
        i_w = torch.sigmoid(torch.unsqueeze(self.w(t_g)) + self.u(h))  # node, hidden
        i_w = F.softmax(i_w, 0)
        # c is calculated according to c
        new_c = f * t_c + torch.sum(c * i_w, 0)
        '''
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class LSTM(nn.Module):

    def __init__(self, emb_size, hidden_size, dropout, bidirec=True):
        super(LSTM, self).__init__()
        self.bidirec = bidirec
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=1, dropout=dropout,
                           bidirectional=bidirec, batch_first=True)

    def forward(self, emb, lengths):
        length, indices = torch.sort(lengths, dim=0, descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = list(torch.unbind(length, dim=0))
        embs = pack(torch.index_select(emb, dim=0, index=indices), input_length, batch_first=True)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs, batch_first=True)[0]
        outputs = torch.index_select(outputs, dim=0, index=ind)
        h = torch.index_select(h, dim=1, index=ind)
        c = torch.index_select(c, dim=1, index=ind)
        # h (directions*layers, batch, hidden)
        if not self.bidirec:
            return outputs, (h, c)
        else:
            batch_size = h.size(1)
            # h (directions*layers, batch, hidden) ---> h (layers, batch, direction*hidden)
            h = h.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.hidden_size)
            c = c.transpose(0, 1).contiguous().view(batch_size, -1, 2 * self.hidden_size)
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state


class RNN_Encoder(nn.Module):
    def __init__(self, hidden_size, emb_size, dropout):
        super(RNN_Encoder, self).__init__()
        self.attn_pooling = Attentive_Pooling(hidden_size)
        self.hidden_size = hidden_size
        self.lstm = LSTM(emb_size, hidden_size // 2, dropout)
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, hidden_size)
        self.attn_u = nn.Linear(hidden_size, 1, bias=False)

    def attention(self, memory, query, mask=None):
        '''

        :param memory: (node, seq, filter_size)
        :param query:   (node, hidden_size)
        :param mask:
        :return:
        '''
        h = torch.tanh(self.attn_v(memory) + torch.unsqueeze(self.attn_w(query), 1))  # node, seq, filter
        score = torch.squeeze(self.attn_u(h), -1)  # node, seq
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node, seq
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, 1)  # node, filter
        return s

    def hierarchical(self, text_emb, mask, lengths, query):
        batch, tweet, seq, embed = text_emb.size()
        x = text_emb.view(-1, seq, embed)
        hidden, _ = self.lstm(x, lengths.view(-1))
        repeat_query = query.repeat(tweet, 1)
        attn_hidden = self.attention(hidden.view(batch * tweet, -1, self.hidden_size), repeat_query)
        # batch*tweet, hidden
        x = attn_hidden.view(batch, tweet, embed)
        tweet_mask = mask[:, :, 0]
        hidden = self.attention(x.view(batch, -1, self.hidden_size) * torch.unsqueeze(tweet_mask, -1), query)

        return hidden

    def forward(self, text_emb, mask, lengths):
        batch, seq, embed = text_emb.size()
        x = text_emb.view(-1, seq, embed)
        output, state = self.lstm(x, lengths)
        return self.attn_pooling(output)


class CNN_Encoder(nn.Module):
    def __init__(self, filter_size, hidden_size):
        super(CNN_Encoder, self).__init__()
        assert filter_size * 3 == hidden_size

        # network configure
        self.num_filter = filter_size  # number of conv1d filters
        self.window_size = [3, 4, 5]  # conv1d kernel window size
        self.nfeat_trans = len(self.window_size) * self.num_filter  # features use concatination of mul and sub

        self.encoder_1 = nn.Conv2d(in_channels=1, out_channels=self.num_filter,
                                   kernel_size=(self.window_size[0], hidden_size))
        self.encoder_2 = nn.Conv2d(in_channels=1, out_channels=self.num_filter,
                                   kernel_size=(self.window_size[1], hidden_size))
        self.encoder_3 = nn.Conv2d(in_channels=1, out_channels=self.num_filter,
                                   kernel_size=(self.window_size[2], hidden_size))

        self.attn_w = nn.Linear(hidden_size, filter_size)
        self.attn_v = nn.Linear(filter_size, filter_size)
        self.attn_u = nn.Linear(filter_size, 1, bias=False)

    def attention(self, memory, query, mask=None):
        '''

        :param memory: (node, seq, filter_size)
        :param query:   (node, hidden_size)
        :param mask:
        :return:
        '''
        h = torch.tanh(self.attn_v(memory) + torch.unsqueeze(self.attn_w(query), 1))  # node, seq, filter
        score = torch.squeeze(self.attn_u(h), -1)  # node, seq
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node, seq
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, 1)  # node, filter
        return s

    def hierarchical(self, text_emb, mask, lengths=None, query=None):
        batch, tweet, seq, embed = text_emb.size()
        # input to conv2d: (N, c_in, h_in, w_in), output: (N, c_out, h_out, w_out)
        x = text_emb.contiguous().view(batch * tweet, 1, seq, embed)
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x)
        x_3 = self.encoder_3(x)
        if query is None:
            x_1 = torch.max(F.relu(x_1), dim=2)[0]
            x_1 = x_1.view(-1, self.num_filter)
            x_2 = torch.max(F.relu(x_2), dim=2)[0]
            x_2 = x_2.view(-1, self.num_filter)
            x_3 = torch.max(F.relu(x_3), dim=2)[0]
            x_3 = x_3.view(-1, self.num_filter)
            # batch*tweet, hidden
            x = torch.cat([x_1, x_2, x_3], 1)
            x = x.view(batch, 1, tweet, embed)
            x_1 = self.encoder_1(x)
            x_2 = self.encoder_2(x)
            x_3 = self.encoder_3(x)
            x_1 = torch.max(F.relu(x_1), dim=2)[0]
            x_1 = x_1.view(-1, self.num_filter)
            x_2 = torch.max(F.relu(x_2), dim=2)[0]
            x_2 = x_2.view(-1, self.num_filter)
            x_3 = torch.max(F.relu(x_3), dim=2)[0]
            x_3 = x_3.view(-1, self.num_filter)
            # batch, hidden
            x = torch.cat([x_1, x_2, x_3], 1)
        else:
            repeat_query = query.repeat(tweet, 1)
            x_1 = self.attention(x_1.view(batch * tweet, -1, self.num_filter), repeat_query)
            x_2 = self.attention(x_2.view(batch * tweet, -1, self.num_filter), repeat_query)
            x_3 = self.attention(x_3.view(batch * tweet, -1, self.num_filter), repeat_query)
            # batch*tweet, emb
            x = torch.cat([x_1, x_2, x_3], 1)
            x = x.contiguous().view(batch, 1, tweet, embed)
            x_1 = self.encoder_1(x)
            x_2 = self.encoder_2(x)
            x_3 = self.encoder_3(x)
            x_1 = self.attention(x_1.view(batch, -1, self.num_filter), query)
            x_2 = self.attention(x_2.view(batch, -1, self.num_filter), query)
            x_3 = self.attention(x_3.view(batch, -1, self.num_filter), query)
            # batch*tweet, hidden
            x = torch.cat([x_1, x_2, x_3], 1)

        return x

    def forward(self, text_emb, text_mask, query=None):
        # index to w2v
        batch, seq, embed = text_emb.size()
        # input to conv2d: (N, c_in, h_in, w_in), output: (N, c_out, h_out, w_out)
        x = text_emb.contiguous().view(batch, 1, seq, embed)
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x)
        x_3 = self.encoder_3(x)
        if query is None:
            x_1 = torch.max(F.relu(x_1), dim=2)[0]
            x_1 = x_1.view(-1, self.num_filter)
            x_2 = torch.max(F.relu(x_2), dim=2)[0]
            x_2 = x_2.view(-1, self.num_filter)
            x_3 = torch.max(F.relu(x_3), dim=2)[0]
            x_3 = x_3.view(-1, self.num_filter)
        else:
            x_1 = self.attention(x_1.view(batch, -1, self.num_filter), query)
            x_2 = self.attention(x_2.view(batch, -1, self.num_filter), query)
            x_3 = self.attention(x_3.view(batch, -1, self.num_filter), query)

        # batch, hidden
        x = torch.cat([x_1, x_2, x_3], 1)
        return x

class TextSLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, relation_num):
        super(TextSLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wh = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 4)
        self.rgcn = RGCN(hidden_size, hidden_size, relation_num)

    def forward(self, x, h, c, g, adjs):
        '''

        :param x:   (node, emb)
            embedding of the node, news and initial node embedding
        :param h:   (node, hidden)
            hidden state from last layer
        :param c:   candidate from last layer
        :param g:   (hidden)
            hidden state of the global node
        :param adj:   (node, node)
            if use RGCN, there should be multiple gcns, each one for a relation
        :return:
        '''
        hn = self.rgcn(h, adjs)
        gates = self.Wh(h) + self.U(x) + self.Wn(hn) + torch.unsqueeze(self.V(g), 0)
        i, f, o, u = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c


class TextGLSTMCell(nn.Module):
    def __init__(self, hidden_size, attn_pooling):
        super(TextGLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.attn_pooling = attn_pooling

    def forward(self, g, c_g, h, c):
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        # h_avg = torch.mean(h, 1)
        h_avg = self.attn_pooling(h)
        f, o = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_size, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), -2) + self.u(h))
        f_w = F.softmax(f_w, -2)
        new_c = f * c_g + torch.sum(c * f_w, -2)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c



class CGM(nn.Module):
    def __init__(self, config, word_vocab):
        super(CGM, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.emb_size = config.emb_size
        self.word_vocab = word_vocab
        if word_vocab.emb is not None:
            self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(word_vocab.emb), freeze=config.freeze_emb)
        else:
            self.word_emb = nn.Embedding(word_vocab.voc_size, config.emb_size)
        self.node_emb = nn.Embedding(config.node_num, config.hidden_size)
        if config.encoder == 'cnn':
            self.text_encoder = CNN_Encoder(config.filter_size, config.hidden_size)
        elif config.encoder == 'rnn':
            self.text_encoder = RNN_Encoder(config.hidden_size, config.emb_size, config.dropout)
        self.feature_weight_price = nn.Linear(6, config.hidden_size)
        self.feature_weight_volume = nn.Linear(2, config.hidden_size)
        self.feature_combine = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # self.last_layer_combin = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.cca_price = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*2), nn.ReLU(), \
                                       nn.Linear(config.hidden_size*2, config.hidden_size*2), nn.ReLU(), \
                                       nn.Linear(config.hidden_size*2, config.hidden_size))
        self.cca_volume = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size*2), nn.ReLU(), \
                                       nn.Linear(config.hidden_size*2, config.hidden_size*2), nn.ReLU(), \
                                       nn.Linear(config.hidden_size*2, config.hidden_size))
        self.attn_pooling = Attentive_Pooling(config.hidden_size)
        self.s_cell = SLSTMCell(config.hidden_size, config.hidden_size, config.relation_num, config.dropout)
        self.g_cell = GLSTMCell(config.hidden_size, self.attn_pooling, config.dropout)
        self.text_s_cell = TextSLSTMCell(config.hidden_size, config.hidden_size, config.relation_num)
        self.text_g_cell = TextGLSTMCell(config.hidden_size, self.attn_pooling)
        self.w_out = nn.Linear(config.hidden_size, 1)
        self.num_layers = config.num_layers
        self.dropout = torch.nn.Dropout(config.dropout)

    def encode(self, span_nodes, node_text, text_mask, text_length, node_feature, adj):
        """

        :param span_nodes: (time, node)
        :param text_nodes: (node)
        :param node_text:   (node, seq)
        :param text_length:
        :param text_mask:  (node, seq)
        :param node_feature:   (time, node, feature_size)
        :param adj:
        :return:
        """
        node_size, seq_size = node_text.size()
        word_emb = self.word_emb(node_text)
        node_emb = self.node_emb(span_nodes)  # idx of node
        text_vector = self.text_encoder(word_emb.view([-1, seq_size, self.emb_size]),
                                        text_mask.view(-1, seq_size), lengths=text_length.view(-1))
        # text_vector = self.node_merge(torch.cat([text_vector, bert_vec.view([-1, 768])], -1))
        x_price = self.feature_weight_price(node_feature[:,:,:6])
        x_volume = self.feature_weight_volume(node_feature[:,:,6:])
        # node_emb = self.node_emb(span_nodes)
        # the role of init_node_vector is the same as the x
        #NOTE price graph
        last_h_time_price = torch.squeeze(x_price[0], 0)  # node feature 
        last_c_time_price = torch.squeeze(x_price[0], 0)  # node feature 
        last_g_time_price = self.attn_pooling(last_h_time_price, node_emb)
        last_c_g_time_price = self.attn_pooling(last_c_time_price, node_emb)
        # h_states = []
        time = node_feature.size(0)
        for t in range(time):
            # init
            last_h_layer_price = last_h_time_price
            last_c_layer_price = last_c_time_price
            # information integration 
            last_g_layer_price, last_c_g_layer_price = self.g_cell.init_forward(last_g_time_price, last_c_g_time_price, last_h_layer_price, last_c_layer_price, node_emb)
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_price, last_c_layer_price = self.s_cell(torch.squeeze(x_price[t], 0), last_h_layer_price, last_c_layer_price, last_g_layer_price, last_h_time_price, adj)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_price, last_c_g_layer_price = self.g_cell(last_g_layer_price, last_c_g_layer_price, last_g_time_price, last_c_g_time_price, last_h_layer_price, last_c_layer_price, node_emb)
            last_h_time_price, last_c_time_price = last_h_layer_price, last_c_layer_price
            last_g_time_price, last_c_g_time_price = last_g_layer_price, last_c_g_layer_price
        #NOTE volume graph 
        last_h_time_volume = torch.squeeze(x_volume[0], 0)  # node feature 
        last_c_time_volume = torch.squeeze(x_volume[0], 0)  # node feature 
        last_g_time_volume = self.attn_pooling(last_h_time_volume, node_emb)
        last_c_g_time_volume = self.attn_pooling(last_c_time_volume, node_emb)
        # h_states = []
        time = node_feature.size(0)
        for t in range(time):
            # init
            last_h_layer_volume = last_h_time_volume
            last_c_layer_volume = last_c_time_volume
            # information integration 
            last_g_layer_volume, last_c_g_layer_volume = self.g_cell.init_forward(last_g_time_volume, last_c_g_time_volume, last_h_layer_volume, last_c_layer_volume, node_emb)
            for l in range(self.num_layers):
                # x, h, c, g, h_t, adj
                last_h_layer_volume, last_c_layer_volume = self.s_cell(torch.squeeze(x_volume[t], 0), last_h_layer_volume, last_c_layer_volume, last_g_layer_volume, last_h_time_volume, adj)
                # g, c_g, t_g, t_c, h, c
                last_g_layer_volume, last_c_g_layer_volume = self.g_cell(last_g_layer_volume, last_c_g_layer_volume, last_g_time_volume, last_c_g_time_volume, last_h_layer_volume, last_c_layer_volume, node_emb)
            last_h_time_volume, last_c_time_volume = last_h_layer_volume, last_c_layer_volume
            last_g_time_volume, last_c_g_time_volume = last_g_layer_volume, last_c_g_layer_volume
        #NOTE text graph 
        node_vector = self.feature_combine(torch.cat([last_h_layer_volume, last_h_layer_price, text_vector, node_emb], -1))
        cca_price, cca_volume = self.cca_price(last_h_time_price), self.cca_volume(last_h_time_volume)
        last_h_layer, last_c_layer, last_g_layer, last_c_g_layer  = last_h_layer_volume, last_c_layer_volume, last_g_layer_volume, last_c_g_layer_volume
        for l in range(self.num_layers):
            last_h_layer, last_c_layer = self.text_s_cell(node_vector, last_h_layer, last_c_layer, last_g_layer, adj)
            # g, c_g, h, c
            last_g_layer, last_c_g_layer = self.text_g_cell(last_g_layer, last_c_g_layer, last_h_layer, last_c_layer)       
        # vec = self.feature_combine(self.dropout(torch.cat([last_h_time, text_vector, node_emb], -1)))

        return last_h_layer, cca_price, cca_volume

    def forward(self, span_nodes, node_text, text_mask, node_feature, adj):
        '''
        :param batch:
            nodes: (time, graph_node)
            node_text: (time, node, seq)
            adjs: (time, node, node)
        :param use_cuda:
        :return:    (node, label)
        '''
        text_lengths = text_mask.sum(-1).int()
        assert text_lengths.max() <= node_text.size(-1) and text_lengths.min() > 0, (text_lengths, node_text.size())
        last_h, cca_price, cca_volume = self.encode(span_nodes, node_text, text_mask, text_lengths, node_feature, adj)
        # the index 0 here is the first time step, which is because that this is the initialization
        return self.w_out(self.dropout(last_h)), cca_price, cca_volume


