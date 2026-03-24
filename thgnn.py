import torch
from torch import nn, optim
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, GRU
from torch.nn import functional as F
from torch_geometric.nn.pool import SAGPooling
# from torch_geometric.nn.aggr import SetTransformerAggregation, MLPAggregation
# from set_transformer_models import SetTransformer
from torch_geometric.utils import dense_to_sparse, sort_edge_index, to_dense_batch
from collections import OrderedDict
from film import Scale_4, Shift_4
import sys


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        # print(x)
        # emb, h = self.lstm(x)
        emb, h = self.gru(x, h)
        # print(out)
        # sys.exit()
        out = self.fc(self.relu(emb))

        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


def gumbel_softmax(logits: torch.tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10,
                   dim: int = -1) -> torch.tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = gumbels.softmax(dim)
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class ReadoutNN_mean(torch.nn.Module):
    def __init__(self, args):
        super(ReadoutNN_mean, self).__init__()
        self.args = args
        # self.fc = nn.Sequential(OrderedDict([
        #                         ('fc1', nn.Linear(args.hid_dim, int(args.hid_dim/2))),
        #                         ('relu1', nn.ReLU(inplace=True)),
        #                         ]))
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(args.hid_dim, args.hid_dim)),
            ('relu1', nn.LeakyReLU()),
        ]))
        # predictor
        self.readout = nn.Linear(args.hid_dim, 1)
        # self.readout = nn.Linear(args.hid_dim, 1)
        self.sigmod = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def forward(self, nodes_embed, adj=None):
        # nodes_embed b*N*d
        nodes_embed = self.fc(nodes_embed)
        mean_emb = torch.mean(nodes_embed, dim=1)
        prediction = self.readout(mean_emb)
        if self.args.sig == 1:
            prediction = self.sigmod(prediction)

        # print('prediction:', prediction)

        return prediction


class TemporalGNN(torch.nn.Module):
    def __init__(self, num_nodes, time_length, hid_dim, input_dim):
        super(TemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.time_length = time_length
        self.hid_dim = hid_dim
        self.input_dim = input_dim

        self.delta = nn.Parameter(torch.ones(*[1], requires_grad=True))
        # self.nerb_attn = nn.Parameter(torch.ones(*[1, time_length, num_nodes, 1], requires_grad=True))
        # torch.nn.init.kaiming_normal_(self.nerb_attn)
        self.softmax4attn = nn.Softmax(dim=2)
        self.self_t = nn.Linear(self.input_dim, hid_dim)
        self.neib_h = nn.Linear(self.input_dim, hid_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.leakyRule = nn.LeakyReLU()  # dimension: b*N*d

    def forward(self, node_feature_t, neib_feature_h, adj):
        # t=num_adjs
        # input: node_feature_t: b*N*1; neib_feature_h:b*N*t; adj: b*t*N*N
        # output: node_emb: b*N*d; neib_emb_timeDecay: b*t*N*d

        # 1. self feauture learning
        self_t_emb = self.self_t(node_feature_t.reshape(-1, self.num_nodes, self.input_dim))  # b*N*d
        self_t_emb = self.dropout(self_t_emb)

        # 2. neighbor feauture learning
        neib_node_feature = neib_feature_h.reshape(-1, self.time_length, self.num_nodes,
                                                   self.input_dim)  # dimension=b*t*N*1
        neib_node_emb = self.neib_h(neib_node_feature)  # b*t*N*d

        # 3. aggregate neighbor's embedding
        aggerate_neib_embed = torch.matmul(adj, neib_node_emb)  # adj:b*t*N*N; aggerate_neib_embed: b*t*N*d

        # 4. apply time decay cofficients on aggerate_neib_embed
        time_decay_lst = torch.tensor([i for i in range(self.time_length - 1, -1, -1)]).to(
            torch.device(self_t_emb.get_device()))
        neib_timeDecay = -torch.mul(time_decay_lst, self.delta)  # dimension= 1*t
        soft_decay = F.softmax(neib_timeDecay, dim=0).reshape(1, self.time_length, 1, 1)  # dimension= 1*t*1*1
        neib_emb_timeDecay = torch.mul(aggerate_neib_embed, soft_decay)  # dimension= b*t*N*d
        neib_t_emb = torch.sum(neib_emb_timeDecay, dim=1)  # dimension: b*N*d

        # 5. sum self embedding and neibor's embedding
        node_emb = self.leakyRule(self_t_emb + neib_t_emb)  # dimension: b*N*d

        return node_emb, aggerate_neib_embed


class CorrelationNN(torch.nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(CorrelationNN, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.2)
        self.encoder = GRUNet(input_dim, self.hidden_dim, (self.hidden_dim) * (args.num_nodes), args.num_rnn_layers)

        if self.args.bottle == 0:
            self.types_funcs = nn.Parameter(torch.ones(*[self.hidden_dim, self.hidden_dim], requires_grad=True))  # 8*16*4, 设置了八组可学习线性层
            torch.nn.init.kaiming_normal_(self.types_funcs)
        # self.b1 = nn.Parameter(torch.zeros(args.typeFunc_dim, requires_grad=True))
        # self.types_funcs = nn.Linear(self.hidden_dim, args.typeFunc_dim)
            if self.args.lm == 1:
                self.layer_norm = nn.LayerNorm([args.num_nodes, self.hidden_dim])

            self.scale = nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)

            self.shift = nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)

        self.relu = nn.ReLU()


        if self.args.bottle == 1:
            self.types_funcs = nn.Parameter(
                torch.ones(*[self.hidden_dim, args.typeFunc_dim], requires_grad=True))  # 8*16*4, 设置了八组可学习线性层
            torch.nn.init.kaiming_normal_(self.types_funcs)

            self.types_funcs_1 = nn.Parameter(
                torch.ones(*[args.typeFunc_dim, self.hidden_dim],
                           requires_grad=True))  # 8*4*8, 设置了八组可学习线性层
            torch.nn.init.kaiming_normal_(self.types_funcs_1)
            # self.types_funcs_1 = nn.Linear(args.typeFunc_dim, 2 * args.typeFunc_dim)
            # self.b2 = nn.Parameter(torch.zeros(2 * args.typeFunc_dim, requires_grad=True))
            if args.lm == 1:
                self.layer_norm = nn.LayerNorm([args.num_nodes, args.typeFunc_dim])
                self.layer_norm_1 = nn.LayerNorm([args.num_nodes, self.hidden_dim])

            self.scale = nn.Linear(self.hidden_dim, (self.hidden_dim) * args.typeFunc_dim + (args.typeFunc_dim) * self.hidden_dim)

            self.shift = nn.Linear(self.hidden_dim, (self.hidden_dim) * args.typeFunc_dim + (args.typeFunc_dim) * self.hidden_dim)


    def get_adjs(self, learned_graph, hard=False):
        # input: node_embeddings: b * t * N * d
        # output: adjs: b*t*N*N.
        node_num = self.args.num_nodes
        learned_graph_stack = torch.stack([learned_graph, 1 - learned_graph], dim=-1)
        adj = gumbel_softmax(learned_graph_stack, tau=1, hard=hard)
        adj = adj[:, :, :, :, 0].clone()

        mask = torch.eye(node_num, node_num).bool().to(adj.get_device())

        adj.masked_fill_(mask, 0)
        adj = torch.mul(adj, learned_graph)
        return adj

    def get_graph(self, node_embeddings):
        node_num = self.args.num_nodes
        node_embeddings_T = \
            node_embeddings.reshape(node_embeddings.shape[0], self.args.time_length, -1, node_num)

        learned_graph = torch.matmul(node_embeddings, node_embeddings_T)

        norm = torch.norm(node_embeddings, p=2, dim=-1, keepdim=True)  # b t N 1
        norm_T = norm.reshape(-1, self.args.time_length, 1, node_num)
        norm = torch.matmul(norm, norm_T)
        learned_graph = learned_graph / norm
        learned_graph = (learned_graph + 1) / 2.  # 余弦相似度求出来后为什么还要加一再除以2？

        return learned_graph

    def learn_types(self, node_embeddings, heter_types):
        # input: node_embeddings: b*t*N*d;  heter_types: b*N*num_types
        # output: learned_graph: b*t*N*N

        scale = self.scale(node_embeddings)
        scale = F.leaky_relu(scale)
        shift = self.shift(node_embeddings)
        shift = F.leaky_relu(shift)

        if self.args.bottle == 0:

            scale_1 = scale[:, :, :, :].reshape(-1, self.args.time_length, self.args.num_nodes,
                                                                                        self.hidden_dim,
                                                                                        self.hidden_dim)
            # scale_2 = scale[:, :, :, self.hidden_dim * self.args.typeFunc_dim:(self.hidden_dim + 1) * self.args.typeFunc_dim]


            shift_1 = shift[:, :, :, :].reshape(-1, self.args.time_length,
                                                                                        self.args.num_nodes,
                                                                                        self.hidden_dim,
                                                                                        self.hidden_dim)
            node_embeddings = \
                node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, 1,
                                        self.hidden_dim)  # 10*30*14*1*16

            typeFuncs_w = (1 + scale_1) * self.types_funcs + shift_1
            # typeFuncs_b = (1+scale_2)*self.b1+shift_2
            typeFuncs_w = typeFuncs_w.permute(0, 1, 2, 4, 3)
            node_embeddings = torch.sum(node_embeddings * typeFuncs_w, dim=-1)
            if self.args.lm == 1:
                node_embeddings = self.layer_norm(torch.squeeze(node_embeddings))  # 10*30*14*8
            node_embeddings = F.leaky_relu(node_embeddings)

        # shift_2 = shift[:, :, :, self.hidden_dim * self.args.typeFunc_dim:(self.hidden_dim + 1) * self.args.typeFunc_dim]

        if self.args.bottle == 1:
            scale_1 = scale[:, :, :, :self.hidden_dim * self.args.typeFunc_dim].reshape(-1, self.args.time_length,
                                                                                        self.args.num_nodes,
                                                                                        self.hidden_dim,
                                                                                        self.args.typeFunc_dim)
            # scale_2 = scale[:, :, :, self.hidden_dim * self.args.typeFunc_dim:(self.hidden_dim + 1) * self.args.typeFunc_dim]

            shift_1 = shift[:, :, :, :self.hidden_dim * self.args.typeFunc_dim].reshape(-1, self.args.time_length,
                                                                                        self.args.num_nodes,
                                                                                        self.hidden_dim,
                                                                                        self.args.typeFunc_dim)

            scale_3 = scale[:, :, :, (self.hidden_dim) * self.args.typeFunc_dim:]\
                .reshape(-1, self.args.time_length, self.args.num_nodes, self.args.typeFunc_dim, self.hidden_dim)
            # scale_4 = scale[:, :, :, (self.hidden_dim + 1) * self.args.typeFunc_dim + self.args.typeFunc_dim * 2 * self.args.typeFunc_dim:]

            shift_3 = shift[:, :, :,
                      (self.hidden_dim) * self.args.typeFunc_dim:]\
                .reshape(-1, self.args.time_length, self.args.num_nodes, self.args.typeFunc_dim, self.hidden_dim)
            # shift_4 = shift[:, :, :,
            #           (self.hidden_dim + 1) * self.args.typeFunc_dim + self.args.typeFunc_dim * 2 * self.args.typeFunc_dim:]
        # 1. apply tyep specific transformation on node_embeddings
            node_embeddings = \
                node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, 1, self.hidden_dim)  # 10*30*14*1*16

            typeFuncs_w = (1+scale_1)*self.types_funcs+shift_1
            # typeFuncs_b = (1+scale_2)*self.b1+shift_2
            typeFuncs_w = typeFuncs_w.permute(0, 1, 2, 4, 3)
            node_embeddings = torch.sum(node_embeddings*typeFuncs_w, dim=-1)

            if self.args.lm == 1:
                node_embeddings = self.layer_norm(torch.squeeze(node_embeddings))  # 10*30*14*8
            node_embeddings = F.leaky_relu(node_embeddings)

        # if self.args.bottle == 1:
            node_embeddings = \
                node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, 1, self.args.typeFunc_dim)

            typeFuncs_w = (1 + scale_3) * self.types_funcs_1 + shift_3
            # typeFuncs_b = (1 + scale_4) * self.b2 + shift_4
            typeFuncs_w = typeFuncs_w.permute(0, 1, 2, 4, 3)
            node_embeddings = torch.sum(node_embeddings * typeFuncs_w, dim=-1)
            if self.args.lm == 1:
                node_embeddings = self.layer_norm_1(torch.squeeze(node_embeddings))

        # print(node_embeddings[0,0,0,:])
        # 2. calculate correlations between nodes
        learned_graph = self.get_graph(node_embeddings)
        # learned_graph = self.get_graph(node_embeddings_1)

        if self.args.film_reg == 1:
            scale_cost = torch.norm(scale, p=2, dim=-1)
            scale_cost = torch.mean(scale_cost)

            shift_cost = torch.norm(shift, p=2, dim=-1)
            shift_cost = torch.mean(shift_cost)

            reg_cost = scale_cost + shift_cost
        else:
            reg_cost = 0

        return learned_graph, reg_cost

    def forward(self, source, heter_types):
        # input: source - b*N*t
        # outpt: adj - b*t*N*N

        # 1. Use GRU to generate node_embeddings
        source = source.reshape(-1, self.args.time_length, self.input_dim)  # source - b*t*N
        init_state = self.encoder.init_hidden(source.shape[0])
        node_embeddings, _ = self.encoder(source, init_state)  # B, T, N*hidden
        node_embeddings = node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, self.hidden_dim)

        # 2. According to heterogenous node types, generate correlations between nodes
        # learned_graph = self.get_graph(node_embeddings) # get graphs only by node's embeddings
        learned_graph, reg_cost = self.learn_types(node_embeddings, heter_types)  # get graph by node's embeddings and types

        # 3. Use Gumble-softmax to generate adj matrices
        adj_soft = F.softmax(learned_graph, dim=-1)
        adj_hard = self.get_adjs(adj_soft, hard=True)  # b*t*N*N

        return adj_hard, reg_cost


class HTGNNModel(torch.nn.Module):
    def __init__(self, args):
        super(HTGNNModel, self).__init__()
        self.args = args
        # block 1
        self.cor_rnn1 = CorrelationNN(args, args.num_nodes, args.hid_dim) # was args.cor_dim
        self.TGNN1 = TemporalGNN(args.num_nodes, args.time_length, args.hid_dim, args.feature_dimension)

        self.readout = ReadoutNN_mean(args)
        # self.readout = ReadoutNN_Agg(args)

    def forward(self, node_feature, adjs1, heter_types, training=True):
        # node_feature: b*N*t, b is the batch size, N is the number of nodes, t is the time length
        node_feature = node_feature.reshape(-1, self.args.num_nodes, self.args.time_length)

        # 1. Get correlations between nodes, i.e., adj matrices
        # Because we refer nodes at each timestamp t form a graph, there are t graphs (t adj matrices).
        adjs1, reg_cost = self.cor_rnn1(node_feature, heter_types)  # b*t*N*N
        # 2. Use temporal GNN to learn the last time graph embedding
        node_feature_t = node_feature[:, :, -1]  # b*N*1
        node_emb_t, _ = \
            self.TGNN1(node_feature_t, node_feature, adjs1)  # node_emb_t: b*N*d; neib_emb_timeDecay: b*t*N*d

        # 3. Predict the label
        pred = self.readout(node_emb_t, adjs1[:, -1, :, :])
        return pred, reg_cost


class GNNModel(torch.nn.Module):  # ablated GNN
    def __init__(self, args):
        super(GNNModel, self).__init__()
        self.args = args
        self.input_dim = args.num_nodes
        self.cor_embed_dim = args.cor_embed_dim
        self.hidden_dim = args.hid_dim
        self.encoder = GRUNet(self.input_dim, self.cor_embed_dim, (self.cor_embed_dim) * (args.num_nodes),
                              args.num_rnn_layers)

        self.linear_encorder = nn.Linear(args.time_length, self.hidden_dim)
        self.linear = nn.Linear(args.time_length, self.hidden_dim)

        self.readout = ReadoutNN_mean(args)
        # self.readout = ReadoutNN_Agg(args)

    def forward(self, node_feature, adjs1, heter_types, training=True):
        # 1. Use Linear to generate node_embeddings
        node_feature = node_feature.reshape(-1, self.input_dim, self.args.time_length)  # source - b*N*t
        node_embeddings = self.linear_encorder(node_feature)  # B, T, N, d
        # node_embeddings = node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, self.cor_embed_dim)

        # 2. create adj
        node_embeddings_T = node_embeddings.reshape(-1, self.hidden_dim, self.args.num_nodes)
        adj = torch.matmul(node_embeddings, node_embeddings_T)  # b*N*N

        # 3. aggreate features
        node_agg = torch.matmul(adj, node_feature)
        graph_emb = self.linear(node_agg)

        # 4. Predict the label
        pred = self.readout(graph_emb)

        return pred

    def forward_GRU(self, node_feature, adjs1, heter_types, training=True):
        # 1. Use GRU to generate node_embeddings
        node_feature = node_feature.reshape(-1, self.input_dim, self.args.time_length)  # source - b*N*t
        init_state = self.encoder.init_hidden(node_feature.shape[0])
        node_embeddings, _ = self.encoder(node_feature, init_state)  # B, T, N*hidden
        node_embeddings = node_embeddings.reshape(-1, self.args.time_length, self.args.num_nodes, self.cor_embed_dim)

        # 2. create adj
        node_embeddings = node_embeddings[:, -1, :, :]  # (B N d), Pick the last output of GRU
        node_embeddings_T = node_embeddings.reshape(-1, self.cor_embed_dim, self.args.num_nodes)
        adj = torch.matmul(node_embeddings, node_embeddings_T)  # b*N*N

        # 3. aggreate features
        node_feature = node_feature.reshape(-1, self.input_dim, self.args.time_length)  # source - b*N*t
        node_agg = torch.matmul(adj, node_feature)
        graph_emb = self.linear(node_agg)

        # 4. Predict the label
        pred = self.readout(graph_emb)

        return pred

