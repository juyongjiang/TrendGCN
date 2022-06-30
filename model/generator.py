import torch
import torch.nn as nn
import torch.nn.functional as F


class MAAGCN(nn.Module):
    def __init__(self, args):
        super(MAAGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.window = args.lag
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(self.window, args.embed_dim), requires_grad=True)
        self.node_flows = nn.Parameter(torch.randn(args.embed_dim, args.embed_dim), requires_grad=True)

        self.encoder = MAGRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim, args.num_layers)

        self.layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(0.1)

        # predictor (if it is adaptive for each node, it maybe better)
        # self.weights = nn.Parameter(torch.FloatTensor(args.embed_dim, args.horizon * self.output_dim, self.hidden_dim, 1)) # [D, T, C, F]
        # self.bias = nn.Parameter(torch.FloatTensor(args.embed_dim, args.horizon * self.output_dim, 1)) # [D, F]

        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, norm_dis_matrix):
        # source: B, T, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, norm_dis_matrix, init_state, self.node_embeddings, self.node_flows, self.time_embeddings) # B, T, N, hidden
        output = self.out_dropout(self.layernorm(output[:, -1:, :, :])) # B, 1, N, hidden
        # output = output[:, -1:, :, :]

        # _, output = self.encoder(source, init_state, self.node_embeddings, self.node_flows, self.time_embeddings) # B, T, N, hidden
        # output = self.out_dropout(self.layernorm(torch.mean(torch.stack(output, dim=1), dim=1).unsqueeze(1)))# output[:, -1:, :, :] # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output)) # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2) # B, T, N, C

        return output


class MAGRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(MAGRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.magrnn_cells = nn.ModuleList()
        self.magrnn_cells.append(GRUCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.magrnn_cells.append(GRUCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, norm_dis_matrix, init_state, node_embeddings, node_flows, time_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1] # T
        current_inputs = x
        output_hidden = []

        # time series similarity
        # seq_sim = nn.Softmax(dim=-1)(torch.mean(torch.einsum("btmc,bmkc->btkc", x.permute(0, 2, 1, 3), x), dim=0))[:, :, 0] # [B, N, T, D] * [B, T, N, D] -> [B, N, N, D]

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length): # one by one for GRU
                state = self.magrnn_cells[i](current_inputs[:, t, :, :], norm_dis_matrix, state, node_embeddings, node_flows, time_embeddings[t]) # [B, N, hidden_dim]
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1) # [B, T, N, D]
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.magrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0) # (num_layers, B, N, hidden_dim)


class GRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(GRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = MAGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, node_num)
        self.update = MAGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, node_num)

    def forward(self, x, norm_dis_matrix, state, node_embeddings, node_flows, time_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) # [B, N, 1+D]
        z_r = torch.sigmoid(self.gate(input_and_state, norm_dis_matrix, node_embeddings, node_flows, time_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, norm_dis_matrix, node_embeddings, node_flows, time_embeddings))
        h = r*state + (1-r)*hc

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class MAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, node_num):
        super(MAGCN, self).__init__()
        self.cheb_k = cheb_k
        self.node_num = node_num
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)) # [D, cheb_k, C, F]
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out)) # [D, F]

        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.embs_dropout = nn.Dropout(0.1)
        self.layernorm_graph = nn.LayerNorm(node_num, eps=1e-12)
    
    def forward(self, x, norm_dis_matrix, node_embeddings, node_flows, time_embeddings):
        # x shaped[B, N, C], node_flows shaped [N, N], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0] 
        # node_time = torch.mul(node_embeddings, time_embeddings.unsqueeze(0)) # [N, D] * [1, D] -> [N, D]
        # x_flow = torch.mean(x, dim=0) # [N, C]
        # x_rel_flow = F.softmax(torch.matmul(x_flow, x_flow.permute(1, 0)), dim=1) # [N, N]
        node_embeddings = self.embs_dropout(self.layernorm(node_embeddings + time_embeddings.unsqueeze(0))) # torch.mul(node_embeddings, node_time)
        # node_flow_sim = self.embs_dropout(self.layernorm(torch.matmul(node_embeddings, node_flows) + time_embeddings.unsqueeze(0)))
        supports = F.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)), dim=1)
        # supports = F.softmax(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)) + norm_dis_matrix, dim=1) #+ F.softmax(torch.mm(node_flow_sim, node_flow_sim.transpose(0, 1)), dim=1) # [N, N]

        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0) # [cheb_k, N, N]

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool) # N, dim_out
        
        x_g = torch.einsum("knm,bmc->bknc", supports, x) # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3) # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias # B, N, dim_out

        return x_gconv