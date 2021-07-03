import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models


class SCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size * 3, hidden_size * 7, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 7, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 7)

    def forward(self, x, h, c, g, c_g):
        gates = self.W(h) + self.U(x) + torch.unsqueeze(self.V(g), 1)
        i, l, r, f, s, o, u = torch.split(gates, self.hidden_size, dim=-1)
        u = torch.tanh(u)
        i, l, r, f, s = torch.unbind(F.softmax(torch.sigmoid(torch.stack([i, l, r, f, s], dim=-2)), -2), dim=-2)
        new_c = c[0] * l + c[1] * f + r * c[2] + s * torch.unsqueeze(c_g, 1) + i * u
        new_h = o * torch.tanh(new_c)
        return new_h, new_c


class GCell(nn.Module):
    def __init__(self, hidden_size):
        super(GCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)

    def forward(self, g, c_g, h, c, mask):
        ''' assume dim=1 is word'''
        h_avg = torch.mean(h, 1)
        f, o = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_size, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), 1) + self.u(h)) - torch.unsqueeze((1 - mask) * 1e16, -1).float()
        f_w = F.softmax(f_w, 1)
        new_c = f * c_g + torch.sum(c * f_w, 1)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class SLSTM(nn.Module):
    def __init__(self, config, vocab):
        super(SLSTM, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if vocab.emb is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(vocab.emb), freeze=config.freeze_emb)
        else:
            self.embedding = nn.Embedding(vocab.voc_size, config.emb_size)
        self.emb_proj = nn.Linear(config.emb_size, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.s_cell = SCell(config.emb_size, config.hidden_size)
        self.g_cell = GCell(config.hidden_size)
        self.w_out = nn.Linear(config.hidden_size, config.label_size)
        self.num_layers = config.num_layers
        self.dropout = torch.nn.Dropout(config.dropout)

    @staticmethod
    def get_hidden_before(hidden_states):
        shape = hidden_states.size()
        start = torch.zeros(shape[0], 1, shape[2]).to(hidden_states.device)
        return torch.cat([start, hidden_states[:, :-1, :]], 1)

    @staticmethod
    def get_hidden_after(hidden_states):
        shape = hidden_states.size()
        end = torch.zeros(shape[0], 1, shape[2]).to(hidden_states.device)
        return torch.cat([hidden_states[:, 1:, :], end], 1)

    def encode(self, word, word_mask):
        word_emb = self.embedding(word)
        init_h_states = self.dropout(self.emb_proj(self.embedding(word)) * torch.unsqueeze(word_mask.float(), -1))
        init_c_states = self.dropout(self.emb_proj(self.embedding(word)) * torch.unsqueeze(word_mask.float(), -1))
        init_g = torch.mean(init_h_states, 1)
        init_c_g = torch.mean(init_c_states, 1)
        for l in range(self.num_layers):
            init_h_states = init_h_states * torch.unsqueeze(word_mask.float(), -1)
            init_c_states = init_c_states * torch.unsqueeze(word_mask.float(), -1)
            new_h_states = torch.cat(
                [self.get_hidden_before(init_h_states), init_h_states, self.get_hidden_after(init_h_states)], -1)
            new_c_states = [self.get_hidden_before(init_c_states), init_c_states, self.get_hidden_after(init_c_states)]
            new_h_states, new_c_states = self.s_cell(word_emb, new_h_states, new_c_states, init_g, init_c_g)
            new_g, new_c_g = self.g_cell(init_g, init_c_g, init_h_states, init_c_states, word_mask)
            init_h_states, init_c_states = new_h_states, new_c_states
            init_g, init_c_g = new_g, new_c_g
        return new_h_states, new_c_states, new_g, new_c_g

    def forward(self, span_nodes, bert_vec, node_text, word_mask, sent_mask, node_feature, adj):
        h_states, c_states, g, c_g = self.encode(node_text, word_mask)
        return self.w_out(self.dropout(g)), self.w_out(torch.mean(g, 0))

