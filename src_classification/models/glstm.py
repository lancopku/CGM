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


def attention(query, key, value, mask=None, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.eq(0), -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # clone linear for 4 times, query, key, value, output
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            assert mask.dim() == 4  # batch, head, seq_len, seq_len
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => head * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


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


class Neighbor_Mean(nn.Module):
    def __init__(self, input_size, hidden_size, position_encoding):
        super(Neighbor_Mean, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.u = nn.Linear(hidden_size, 1)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.position_encoding = position_encoding

    def forward(self, x, h, g, neighbor_index, neighbor_mask, rp_neighbor_index=None):
        '''
        :param x: (batch, sent, emb)
        :param h: (batch, sent, hidden)
        :param g: (batch, hidden)
        :param neighbor_index: (batch, sent, neighbor)
        :param neighbor_mask: (batch, sent, neighbor)
        :return:
        '''
        shape = neighbor_index.size()
        if rp_neighbor_index is not None:
            pos_emb = self.position_encoding(rp_neighbor_index.long())
        else:
            pos_emb = self.position_encoding(neighbor_index.long())
        new_h = torch.cat([torch.unsqueeze(torch.zeros_like(g), 1), h], 1)
        neighbor_index = neighbor_index.view(shape[0], -1)
        ind = torch.unsqueeze(torch.arange(shape[0]), 1).repeat(1, shape[1] * shape[2])
        new_h = new_h[ind.long(), neighbor_index.long()].view(shape[0], shape[1], shape[2], -1) + pos_emb
        neighbors = self.Wn(new_h) * torch.unsqueeze(neighbor_mask.float(), -1)
        # batch, sent, neighbor, hidden
        # note that here index starts from 1 so that the position encoding of h and neighbor are compatible
        hn = torch.mean(neighbors, 2)
        return hn


class Neighbor_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, position_encoding):
        super(Neighbor_Attn, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.u = nn.Linear(hidden_size, 1)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.position_encoding = position_encoding

    def forward(self, x, h, g, neighbor_index, neighbor_mask, rp_neighbor_index=None):
        '''
        :param x: (batch, sent, emb)
        :param h: (batch, sent, hidden)
        :param g: (batch, hidden)
        :param neighbor_index: (batch, sent, neighbor)
        :param neighbor_mask: (batch, sent, neighbor)
        :return:
        '''
        shape = neighbor_index.size()
        if rp_neighbor_index is not None:
            pos_emb = self.position_encoding(rp_neighbor_index.long())
        else:
            pos_emb = self.position_encoding(neighbor_index.long())
        new_h = torch.cat([torch.unsqueeze(torch.zeros_like(g), 1), h], 1)
        neighbor_index = neighbor_index.view(shape[0], -1)
        ind = torch.unsqueeze(torch.arange(shape[0]), 1).repeat(1, shape[1] * shape[2])
        new_h = new_h[ind.long(), neighbor_index.long()].view(shape[0], shape[1], shape[2], -1) + pos_emb
        neighbors = self.Wn(new_h) * torch.unsqueeze(neighbor_mask.float(), -1)
        # batch, sent, neighbor, hidden
        # note that here index starts from 1 so that the position encoding of h and neighbor are compatible
        if rp_neighbor_index is not None:
            s = torch.unsqueeze(self.Wh(h) + self.U(x) + torch.unsqueeze(self.V(g), 1), 2) + neighbors
        else:
            h_pos_emb = self.position_encoding(torch.unsqueeze(1 + torch.arange(shape[1]).to(x.device), 0)).repeat(
                shape[0], 1, 1)
            s = torch.unsqueeze(self.Wh(h + h_pos_emb) + self.U(x) + torch.unsqueeze(self.V(g), 1), 2) + neighbors
        score = torch.squeeze(self.u(s), -1)
        score = F.softmax(score.masked_fill(neighbor_mask.eq(0), -1e9), -1)
        hn = torch.sum(torch.unsqueeze(score, -1) * neighbors, 2)
        return hn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, input, adj):
        support = self.linear(input)
        output = self.activation(torch.sparse.mm(adj, support))
        return output


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
    def __init__(self, input_size, hidden_size, relation_num):
        super(SLSTMCell, self).__init__()
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


class SSumCell(nn.Module):
    def __init__(self, input_size, hidden_size, position_encoding):
        super(SSumCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wh = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 4)
        self.neighbor_attn = Neighbor_Attn(input_size, hidden_size, position_encoding)

    def forward(self, x, h, c, g, neighbor_index, neighbor_mask, rp_neighbor_index=None):
        hn = self.neighbor_attn(x, h, g, neighbor_index, neighbor_mask, rp_neighbor_index)
        new_h = hn + x + h + torch.unsqueeze(g, 1)
        new_c = hn + x + c + torch.unsqueeze(g, 1)
        return new_h, new_c


class SGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.U = nn.Linear(input_size + hidden_size, hidden_size * 2, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 2)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.neighbor_attn = Neighbor_Attn(input_size, hidden_size)

    def forward(self, x, h, g, neighbor_index, neighbor_mask):
        hn = self.neighbor_attn(x, h, g, neighbor_index, neighbor_mask)
        x = torch.cat([x, hn], -1)
        gates = torch.sigmoid(self.W(h) + self.U(x) + torch.unsqueeze(self.V(g), 1))
        z, r = torch.split(gates, self.hidden_size, dim=-1)
        ht = torch.tanh(self.w(r * h) + self.u(x) + torch.unsqueeze(self.v(g), 1))
        h = (1 - z) * h + z * ht
        return h


class GLSTMCell(nn.Module):
    def __init__(self, hidden_size, attn_pooling):
        super(GLSTMCell, self).__init__()
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


class GSumCell(nn.Module):
    def __init__(self, hidden_size, attn_pooling):
        super(GSumCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.attn_pooling = attn_pooling

    def forward(self, g, c_g, h, c, mask):
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        h_avg = self.attn_pooling(h)
        new_c = c_g + h_avg
        new_g = g + h_avg
        return new_g, new_c


class GGRUCell(nn.Module):
    def __init__(self, hidden_size):
        super(GGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, g, h, mask):
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        # h_avg = torch.mean(h, 1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), 1) + self.u(h)) * torch.unsqueeze(mask, -1).float()
        f_w = F.softmax(f_w, 1)
        # h_avg = self.attn_pooling(h)
        h_avg = torch.sum(h * f_w, 1)
        z, r = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_size, dim=-1)
        gt = torch.tanh(self.V(torch.cat([r * g, h_avg], -1)))
        h = (1 - z) * g + z * gt
        return h


class GLSTM(nn.Module):
    def __init__(self, config, word_vocab):
        super(GLSTM, self).__init__()
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
            self.text_encoder = models.CNN_Encoder(config.filter_size, config.hidden_size)
        elif config.encoder == 'rnn':
            self.text_encoder = models.RNN_Encoder(config.hidden_size, config.emb_size, config.dropout)
        self.feature_weight = nn.Linear(config.feature_size, config.hidden_size)
        self.feature_lstm = models.LSTM(config.hidden_size, config.hidden_size, config.dropout, bidirec=False)
        self.feature_combine = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.attn_pooling = Attentive_Pooling(config.hidden_size)
        self.s_cell = SLSTMCell(config.hidden_size, config.hidden_size, config.relation_num)
        self.g_cell = GLSTMCell(config.hidden_size, self.attn_pooling)
        self.w_out = nn.Linear(config.hidden_size, config.label_size)
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
        text_vector = self.text_encoder(word_emb.view([-1, seq_size, self.emb_size]),
                                        text_mask.view(-1, seq_size), lengths=text_length.view(-1))
        # text_vector = self.node_merge(torch.cat([text_vector, bert_vec.view([-1, 768])], -1))
        feature_emb = self.feature_weight(node_feature.transpose(0, 1))
        feature_length = torch.Tensor([node_feature.size(0)]).repeat(node_feature.size(1)).to(node_feature.device)
        _, states = self.feature_lstm(feature_emb, lengths=feature_length)
        feature_vector = states[0].squeeze(0)
        node_emb = self.node_emb(span_nodes)
        node_vector = self.feature_combine(torch.cat([feature_vector, text_vector, node_emb], -1))
        h_states = []
        last_h_layer = node_vector
        last_c_layer = node_vector
        last_g_layer = self.attn_pooling(last_h_layer)  
        last_c_g_layer = self.attn_pooling(last_c_layer)
        for l in range(self.num_layers):
            # x, h, c, g, adj
            last_h_layer, last_c_layer = self.s_cell(node_vector, last_h_layer, last_c_layer, last_g_layer, adj)
            # g, c_g, h, c
            last_g_layer, last_c_g_layer = self.g_cell(last_g_layer, last_c_g_layer, last_h_layer, last_c_layer)
            h_states.append(last_h_layer)
        return last_h_layer, node_vector

    def forward(self, span_nodes, node_text, text_mask, node_feature, adj):
        '''
        :param batch:  time means batch size.  
            nodes: (time, graph_node)
            node_text: (time, node, seq)
            adjs: (time, node, node)
        :param use_cuda:
        :return:    (node, label)
        '''
        text_lengths = text_mask.sum(-1).int()
        assert text_lengths.max() <= node_text.size(-1) and text_lengths.min() > 0, (text_lengths, node_text.size())
        output, node_vectors = self.encode(span_nodes, node_text, text_mask, text_lengths, node_feature, adj)
        # the index 0 here is the first time step, which is because that this is the initialization
        return self.w_out(node_vectors)
