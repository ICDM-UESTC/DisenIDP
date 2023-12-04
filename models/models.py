import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from Optim import ScheduledOptim

from models.TransformerBlock import *
from models.ConvBlock import *

'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

class fusion(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


class LSTMGNN(nn.Module):
    def __init__(self, hypergraphs, args, dropout=0.2):
        super(LSTMGNN, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs) 
        self.win_size = 5

        #hypergraph
        self.H_Item = hypergraphs[0]   
        self.H_User =hypergraphs[1]

        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)

        ### channel self-gating parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### channel self-supervised parameters
        self.ssl_weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        #sequence model
        self.past_gru = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first= True)
        self.past_lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)

        # multi-head attention
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)

        self.future_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, is_FFN=False,
                                                 is_future=True, attn_dropout=dropout)

        self.long_term_att = Long_term_atention(input_size=self.emb_size, attn_dropout=dropout)
        self.short_term_att = Short_term_atention(input_size=self.emb_size, attn_dropout=dropout)
        self.conv = ConvBlock(n_inputs=self.emb_size, n_outputs=self.emb_size, kernel_size=self.win_size, padding = self.win_size-1)
        self.linear = nn.Linear(self.emb_size*3, self.emb_size)

        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def self_supervised_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.ssl_weights[channel]) + self.ssl_bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def hierarchical_ssl(self, em, adj):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        user_embeddings = em
        edge_embeddings = torch.sparse.mm(adj, em)

        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        # Global MIM
        graph = torch.mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def seq2seq_ssl(self, L_fea1, L_fea2, S_fea1, S_fea2):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), -1)

        # Local MIM
        pos = score(L_fea1, L_fea2)
        neg1 = score(L_fea1, S_fea2)
        neg2 = score(L_fea2, S_fea1)
        loss1 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        pos = score(S_fea1, S_fea2)
        loss2 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        return loss1 + loss2

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, H_Time=None, H_Item=None, H_User=None):

        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_rate)
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_rate)
        else:
            H_Item = self.H_Item
            H_User = self.H_User

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)

        all_emb_c2 = [u_emb_c2]
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            # Channel Item
            u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings2]

        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)
        u_emb_c3 = torch.stack(all_emb_c3, dim=1)
        u_emb_c3 = torch.sum(u_emb_c3, dim=1)

        # aggregating channel-specific embeddings
        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)

        return high_embs

    def forward(self, input, label):

        mask = (input == 0)
        mask_label = (label == 0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        ####long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask= mask.cuda())

        ####short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)

        ####future cascade
        future_embs = F.embedding(label, HG_Uemb)
        future_output = self.future_multi_att(future_embs, future_embs, future_embs, mask=mask_label.cuda())
        ####future cascade
        short_emb = self.conv(cas_seq_emb)

        '''SSL loss'''
        graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_Item)
        graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_User)

        seq_ssl_loss = self.seq2seq_ssl(L_cas_emb, future_output, S_cas_emb, short_emb)
    
        '''Prediction'''
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss, seq_ssl_loss

    def model_prediction(self, input):

        mask = (input == 0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed()

        '''cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        ####long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())

        ####short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def model_prediction2(self, input, input_len, HG_Time, HG_Item, HG_User):

        mask = (input == 0)

        b = input_len.size(0)

        '''structure embeddding'''
        HG_Uemb = self.structure_embed(H_Time=HG_Time, H_Item=HG_Item, H_User=HG_User)

        '''cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)

        ####long-term temporal influence
        source_emb = cas_seq_emb[:, 0, :]
        L_cas_emb = self.long_term_att(source_emb, cas_seq_emb, cas_seq_emb, mask=mask.cuda())

        ####short-term temporal influence
        user_cas_gru, _ = self.past_gru(cas_seq_emb)
        user_cas_lstm, _ = self.past_lstm(cas_seq_emb)
        S_cas_emb = self.short_term_att(user_cas_gru, user_cas_lstm, user_cas_lstm, mask=mask.cuda())

        LS_cas_emb = torch.cat([cas_seq_emb, L_cas_emb, S_cas_emb], -1)
        LS_cas_emb = self.linear(LS_cas_emb)

        output = self.past_multi_att(LS_cas_emb, LS_cas_emb, LS_cas_emb, mask)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()



