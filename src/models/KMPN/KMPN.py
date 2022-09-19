"""
KMPN.py:  Model class
Modified from the Pytorch implementation of KGIN
"""

__author__ = "Weizhe Lin"
__copyright__ = "Copyright 2021, Weizhe Lin"
__version__ = "1.0.0"
__email__ = "wl356@cam.ac.uk"
__status__ = "Published for Github"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from transformers import BertConfig
import numpy as np

def check_module_setting(setting, config):
    if setting in config.model_config.module_type:
        return True
    else:
        return False

class Aggregator(nn.Module):
    """
    Convolution aggregator
    """
    def __init__(self, config, n_users, n_factors):
        super(Aggregator, self).__init__()
        self.config = config
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att, relation_edge_weight=None):
        """Forward function of aggregator

        Args:
            entity_emb (tensor): n_entities x n_channel
            user_emb (tensor): n_users x n_channel
            latent_emb (tensor): [description]
            edge_index (tensor): [description]
            edge_type (tensor): [description]
            interact_mat (tensor): [description]
            weight (tensor): [description]
            disen_weight_att (tensor): [description]

        Returns:
            [type]: [description]
        """
        n_entities = entity_emb.shape[0] # n_entities
        channel = entity_emb.shape[1] # emb_size
        emb_size = channel
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index # n_edges, n_edges
        n_edges = head.shape[0]
        
        # For each edge, compute the edge feature from tail embeddings and relation embeddings
        
        ####### check arguments ########
        MATRIX_RELATION_EMB = check_module_setting('MATRIX_RELATION_EMB', self.config)
        INDEPENDENT_RELATION_EMB = check_module_setting('INDEPENDENT_RELATION_EMB', self.config)
        GATED_ATT = check_module_setting('GATED_ATT', self.config)
        
        GATED_ATT_AND_INDEPENDENT_RELATION_EMB = INDEPENDENT_RELATION_EMB and GATED_ATT
        PREFERENCE_EMB = check_module_setting('PREFERENCE_EMB', self.config)
        ####### run ########
        if INDEPENDENT_RELATION_EMB:
            relation_edge_weight_to_be_used = relation_edge_weight
        else:
            relation_edge_weight_to_be_used = weight


        if MATRIX_RELATION_EMB:
            step = min(n_edges // emb_size, 10000)
            final_neigh_relation_emb = []
            for start in range(0, n_edges, step):
                end = min(n_edges, start+step)
                # print(start, end)
                # n_edges x emb_size x emb_size
                edge_relation_emb = relation_edge_weight[edge_type[start:end] - 1]
                # n_edges x 1 x emb_size = (n_edges x 1 x emb_size) matmul (n_edges x emb_size x emb_size)
                # print(entity_emb[head[start:end]][:, None, :].shape, edge_relation_emb.shape)
                mat_res = torch.matmul(entity_emb[head[start:end]][:, None, :], edge_relation_emb)
                # print(mat_res.shape)
                edge_relation_emb = mat_res[:, 0, :]
                # print(edge_relation_emb.shape)
                # n_edges x emb_size = (n_edges x emb_size) * (n_edges x emb_size)
                neigh_relation_emb = entity_emb[tail[start:end]] * edge_relation_emb 
                final_neigh_relation_emb.append(neigh_relation_emb)
            final_neigh_relation_emb = torch.cat(final_neigh_relation_emb, dim=0)

        # elif INDEPENDENT_RELATION_EMB:
        #     #### Using independent edge weights for feature aggregation
        #     # n_edges x emb_size
        #     edge_relation_emb = relation_edge_weight_to_be_used[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        #     # n_edges x emb_size = (n_edges x emb_size) * (n_edges x emb_size)
        #     final_neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
            
        elif GATED_ATT:
            # # n_edges x emb_size
            # edge_relation_emb = relation_edge_weight_to_be_used[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
            # # n_edges = diag [(n_edges x emb_size) matmul (n_edges x emb_size).T]
            # edge_attention = torch.sum(entity_emb[head]*edge_relation_emb, dim=1)
            # print(edge_attention.shape)
            # edge_attention = F.sigmoid(edge_attention).unsqueeze(-1)
            # print(edge_attention.shape)
            # # n_edges x emb_size = (n_edges x emb_size) * (n_edges x emb_size)
            # final_neigh_relation_emb = edge_attention * (entity_emb[tail] * edge_relation_emb)  # [-1, channel]
            # final_neigh_relation_emb = neigh_relation_emb
            # step = n_edges // emb_size
            # print('======', step)
            step = min(n_edges // emb_size, 10000)
            final_neigh_relation_emb = []
            for start in range(0, n_edges, step):
                end = min(n_edges, start+step)
                # n_edges x emb_size
                edge_relation_emb = relation_edge_weight_to_be_used[edge_type[start:end] - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
                # n_edges = diag [(n_edges x emb_size) matmul (n_edges x emb_size).T]
                edge_attention = torch.sum(entity_emb[head[start:end]]*edge_relation_emb, dim=1)
                # print(edge_attention.shape)
                edge_attention = F.sigmoid(edge_attention).unsqueeze(-1)
                # print(edge_attention.shape)
                # n_edges x emb_size = (n_edges x emb_size) * (n_edges x emb_size)
                neigh_relation_emb = edge_attention * (entity_emb[tail[start:end]] * edge_relation_emb)  # [-1, channel]
                # print(neigh_relation_emb.shape)
                final_neigh_relation_emb.append(neigh_relation_emb)

            final_neigh_relation_emb = torch.cat(final_neigh_relation_emb, dim=0)
        else:
            # n_edges x emb_size
            edge_relation_emb = relation_edge_weight_to_be_used[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
            # n_edges x emb_size = (n_edges x emb_size) * (n_edges x emb_size)
            final_neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        
        # indexed by head (n_edges), features are aggregated to each entity
        entity_agg = scatter_mean(src=final_neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        # n_users x n_factors = (n_users x emb_size) matmul (n_factors x emb_size)^T
        score_ = torch.mm(user_emb, latent_emb.t())
        # n_users x n_factors x 1
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        ##########  
        # n_users x emb_size = (n_users x n_entities) sparse matmul (n_entities x emb_size)
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        
        if PREFERENCE_EMB:
            # Train preference emb directly
            # n_factors x emb_size
            # expand to n_users x n_factors x emb_size
            disen_weight = weight.expand(n_users, n_factors, channel)
            # print(disen_weight.shape)
        else:
            # n_factors x emb_size = (n_factors x n_relations - 1) matmul (n_relations - 1 x emb_size)
            # expand to n_users x n_factors x emb_size
            disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                    weight).expand(n_users, n_factors, channel)
        # for each user compute the summed emb
        # n_users x emb_size = sum (dim=1) [(n_users x n_factors x emb_size) * (n_users x n_factors x 1)]
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, config, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.config = config
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.num_meta_preferences = config.model_config.num_meta_preferences
        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        if check_module_setting('MATRIX_RELATION_EMB', self.config):
            relation_edge_weight = initializer(torch.empty(n_relations - 1, channel, channel))
            self.relation_edge_weight = nn.Parameter(relation_edge_weight)
        elif check_module_setting('INDEPENDENT_RELATION_EMB', self.config):
            relation_edge_weight = initializer(torch.empty(n_relations - 1, channel))
            self.relation_edge_weight = nn.Parameter(relation_edge_weight)
        else:
            self.relation_edge_weight = None

        PREFERENCE_EMB = check_module_setting('PREFERENCE_EMB', self.config)
        if PREFERENCE_EMB:
            # Train preference emb directly
            weight = initializer(torch.empty(n_factors, channel))
        else:
            weight = initializer(torch.empty(self.num_meta_preferences, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, self.num_meta_preferences))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(config=config, n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def _cul_cor(self):
        SOFT_COS_DISTANCE_COR = check_module_setting('SOFT_COS_DISTANCE_COR', self.config)
        PCA_DISTANCE_COR = check_module_setting('PCA_DISTANCE_COR', self.config)

        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            dcorr = dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
            
            if SOFT_COS_DISTANCE_COR:
                if dcorr>=0.2:
                    softed_dcorr = dcorr
                else:
                    softed_dcorr = (0.5 - 0.5 * torch.cos(np.pi/0.2 * dcorr)) * dcorr
                return softed_dcorr
            else:
                return dcorr

        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        def PCA_svd(X, k, center=True):
            """
            This function performs Primary Component Analysis on X, and return the top k components.
            """
            n = X.size()[0]
            ones = torch.ones(n).view([n,1])
            h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
            H = torch.eye(n) - h
            H = H.cuda()
            X_center =  torch.mm(H.double(), X.double())
            u, s, v = torch.svd(X_center)
            components  = v[:k].t()
            #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
            return components

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            disen_weight_att = self.disen_weight_att
            if PCA_DISTANCE_COR:
                disen_weight_att = PCA_svd(disen_weight_att, int(disen_weight_att.shape[1] * self.config.model_config.pca_ratio))
            
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(disen_weight_att[i], disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(disen_weight_att[i], disen_weight_att[j])
        
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):
        
        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]

        cor = self._cul_cor()

        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att,
                                                 relation_edge_weight=self.relation_edge_weight)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            
        return entity_res_emb, user_res_emb, cor




class Recommender(nn.Module):
    def __init__(self, global_config, data_config, graph, adj_mat, wl_role_data=None, bert_emb_data=None, custom_forward=False):
        super(Recommender, self).__init__()
        self.config = global_config
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = self.config.model_config.l2
        self.sim_decay = self.config.model_config.sim_regularity
        self.cross_system_loss_decay = self.config.model_config.cross_system_loss_decay
        
        self.emb_size = self.config.model_config.dim
        self.context_hops = self.config.model_config.context_hops
        self.n_factors = self.config.model_config.n_factors
        self.node_dropout = self.config.model_config.node_dropout
        self.node_dropout_rate = self.config.model_config.node_dropout_rate
        self.mess_dropout = self.config.model_config.mess_dropout
        self.mess_dropout_rate = self.config.model_config.mess_dropout_rate
        self.ind = self.config.model_config.ind
        self.device = self.config.device
        self.custom_forward = custom_forward

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        

        initializer = nn.init.xavier_uniform_

        if 'WL_EMB' in self.config.model_config.module_type:
            self.wl_role_data = wl_role_data
            self.max_wl_role_index = self.wl_role_data['max_role_index']
            self.wl_role_matrix = torch.LongTensor([i for i in self.wl_role_data['results'].values()])
        
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        
        self.gcn = self._init_model()

        NRMS_BERT_EMB = check_module_setting('NRMS_BERT_EMB', self.config)
        if NRMS_BERT_EMB:
            print(bert_emb_data.keys())
            self.item_bert_emb = bert_emb_data['item_bert_emb']
            self.user_bert_emb = bert_emb_data['user_bert_emb']
            self.bert_emb_size = bert_emb_data['item_bert_emb'].shape[-1]


    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))
        if 'WL_EMB' in self.config.model_config.module_type:
            self.wl_role_emb = initializer(torch.empty(self.max_wl_role_index, self.emb_size))
        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(config=self.config,
                        channel=self.emb_size,
                        n_hops=self.context_hops,
                        n_users=self.n_users,
                        n_relations=self.n_relations,
                        n_factors=self.n_factors,
                        interact_mat=self.interact_mat,
                        ind=self.ind,
                        node_dropout_rate=self.node_dropout_rate,
                        mess_dropout_rate=self.mess_dropout_rate)
        

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def custom_forward_fn(self, batch=None):
        """This function returns generated gcn embeddings instead of only loss

        Args:
            batch ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        # LATE_FUSION = check_module_setting('LATE_FUSION', self.config)
        # if not LATE_FUSION:
        #     item_emb, user_emb = self.aggregate_features(entity_emb=item_emb, user_emb=user_emb)
        
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        
        self.edge_index = self.edge_index.to(user_emb.device)
        self.interact_mat = self.interact_mat.to(user_emb.device)
        
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
         # Late fusion
        # if LATE_FUSION:
        #     entity_gcn_emb, user_gcn_emb = self.aggregate_features(entity_emb=entity_gcn_emb, user_emb=user_gcn_emb)
        
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e, cor), u_e, pos_e, neg_e
    
    def forward(self, batch=None):
        if self.custom_forward:
            return self.custom_forward_fn(batch=batch)
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]


        """
        Note:
            LATE_FUSION is used to combine features from KMPN and NRMS directly using traditional approaches
            such as concatenation, attention pooled sum, and so on.
            However, none of these methods can achieve higher accuracy than pure KMPN models.
            Cross-system contrastive learning was proposed to save this failure, 
                and achieved very good performance while preserving the KMPN's power.
            I decided not to remove these experimental codes such that others interested in
                can try on their own. But for now relevant parts were commented out.
        """
        # LATE_FUSION = check_module_setting('LATE_FUSION', self.config)
        # if not LATE_FUSION:
        #     item_emb, user_emb = self.aggregate_features(entity_emb=item_emb, user_emb=user_emb)
        
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        
        self.edge_index = self.edge_index.to(user_emb.device)
        self.interact_mat = self.interact_mat.to(user_emb.device)
        
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        
        # Late fusion
        # if LATE_FUSION:
        #     entity_gcn_emb, user_gcn_emb = self.aggregate_features(entity_emb=entity_gcn_emb, user_emb=user_gcn_emb)
        
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        total_loss, total_mf_loss, total_emb_loss, total_cor_loss, total_cor = self.create_bpr_loss(u_e, pos_e, neg_e, cor)
        
        NRMS_BERT_EMB = check_module_setting('NRMS_BERT_EMB', self.config)
        NRMS_BERT_EMB_CROSS_SYSTEM = check_module_setting('NRMS_BERT_EMB_CROSS_SYSTEM', self.config)
        NRMS_BERT_EMB_CROSS_SIM = check_module_setting('NRMS_BERT_EMB_CROSS_SIM', self.config)
        
        if NRMS_BERT_EMB:
            self.item_bert_emb = self.item_bert_emb.to(user_emb.device)
            self.user_bert_emb = self.user_bert_emb.to(user_emb.device)
            entity_gcn_bert_emb = self.item_bert_emb
            user_gcn_bert_emb = self.user_bert_emb

            if NRMS_BERT_EMB_CROSS_SYSTEM:
                # Cross system scoring
                u_e = user_gcn_emb[user]
                pos_e, neg_e = entity_gcn_bert_emb[pos_item], entity_gcn_bert_emb[neg_item]
                dual_loss, dual_mf_loss, dual_emb_loss, dual_cor_loss, dual_cor = self.create_bpr_loss(u_e, pos_e, neg_e, 0)
                total_loss += dual_mf_loss * self.cross_system_loss_decay
                # print('add dual_mf_loss', dual_mf_loss * self.cross_system_loss_decay)
                u_e = user_gcn_bert_emb[user]
                pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
                dual_loss, dual_mf_loss, dual_emb_loss, dual_cor_loss, dual_cor = self.create_bpr_loss(u_e, pos_e, neg_e, 0)
                total_loss += dual_mf_loss * self.cross_system_loss_decay
                # print('add dual_mf_loss', dual_mf_loss * self.cross_system_loss_decay)
            
            if NRMS_BERT_EMB_CROSS_SIM:
                # Cross system similarity
                # Proved to be harmful to KMPN training
                def CosineSimilarity(tensor_1, tensor_2):
                    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
                    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
                    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1) ** 2  # no negative

                u_e = user_gcn_emb[user]
                pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
                
                u_e_cross = user_gcn_bert_emb[user]
                pos_e_cross, neg_e_cross = entity_gcn_bert_emb[pos_item], entity_gcn_bert_emb[neg_item]

                sim_user = 1-CosineSimilarity(u_e, u_e_cross)
                sim_pos = 1-CosineSimilarity(pos_e,  pos_e_cross)
                sim_neg = 1-CosineSimilarity(neg_e, neg_e_cross)
                sim = torch.mean(torch.cat([sim_user, sim_pos, sim_neg]))
                sim_loss = sim * self.cross_system_loss_decay
                total_loss += sim_loss
        
        return total_loss, total_mf_loss, total_emb_loss, total_cor

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # LATE_FUSION = check_module_setting('LATE_FUSION', self.config)
        # if not LATE_FUSION:
        #     item_emb, user_emb = self.aggregate_features(entity_emb=item_emb, user_emb=user_emb)
        entity_gcn_emb, user_gcn_emb, cor =  self.gcn(user_emb,
                                                item_emb,
                                                self.latent_emb,
                                                self.edge_index,
                                                self.edge_type,
                                                self.interact_mat,
                                                mess_dropout=False, node_dropout=False)
        # Late fusion
        # if LATE_FUSION:
        #     entity_gcn_emb, user_gcn_emb = self.aggregate_features(entity_emb=entity_gcn_emb, user_emb=user_gcn_emb)
        
        return entity_gcn_emb, user_gcn_emb

    def aggregate_features(self, entity_emb, user_emb):
        """This function performs feature aggregation with graph features and additional features

        Args:
            entity_emb (Tensor): [description]
            user_emb (Tensor): [description]

        Returns:
            Tensor: entity_emb after aggregation
            Tenosr: user_emb after aggregation
        """
        # By default, return the same features
        # final_entity_emb = entity_emb
        # final_user_emb = user_emb

        # Start processing BERT features
        BERT_EMB = check_module_setting('BERT_EMB', self.config)
        BERT_EMB_SUM = check_module_setting('BERT_EMB_SUM', self.config)
        BERT_EMB_ATT_SUM = check_module_setting('BERT_EMB_ATT_SUM', self.config)
        BERT_EMB_ATT_SUM_V2 = check_module_setting('BERT_EMB_ATT_SUM_V2', self.config)
        BERT_EMB_NORM = check_module_setting('BERT_EMB_NORM', self.config)
        BERT_EMB_MUL = check_module_setting('BERT_EMB_MUL', self.config)

        NRMS_BERT_EMB = check_module_setting('NRMS_BERT_EMB', self.config)
        NRMS_BERT_EMB_ATT_SUM = check_module_setting('NRMS_BERT_EMB_ATT_SUM', self.config)
        NRMS_BERT_EMB_ATT_SUM_V2 = check_module_setting('NRMS_BERT_EMB_ATT_SUM_V2', self.config)
        NRMS_BERT_EMB_INIT = check_module_setting('NRMS_BERT_EMB_INIT', self.config)
        NRMS_BERT_EMB_INIT_ADD = check_module_setting('NRMS_BERT_EMB_INIT_ADD', self.config)
        
        if NRMS_BERT_EMB:
            self.item_bert_emb = self.item_bert_emb.to(user_emb.device)
            self.user_bert_emb = self.user_bert_emb.to(user_emb.device)
            transformed_bert_emb = self.item_bert_emb
            
            if NRMS_BERT_EMB_INIT:
                user_emb = self.user_bert_emb
                entity_emb = torch.cat([self.item_bert_emb,
                            entity_emb[self.n_items:, :]], axis=0)
            
            if NRMS_BERT_EMB_INIT_ADD:
                user_emb = user_emb + self.user_bert_emb
                entity_emb = torch.cat([self.item_bert_emb + entity_emb[:self.n_items, :],
                            entity_emb[self.n_items:, :]], axis=0)
            
            if NRMS_BERT_EMB_ATT_SUM:
                # n_items x 128
                merged_item_emb = torch.cat([entity_emb[:self.n_items], transformed_bert_emb], dim=-1)
                att_item = F.softmax(self.att_linear_item(merged_item_emb))
                merged_user_emb = torch.cat([user_emb, self.user_bert_emb], dim=-1)
                att_user = F.softmax(self.att_linear_user(merged_user_emb))
                only_item_embed = entity_emb[:self.n_items] * att_item[:, [0]] + transformed_bert_emb * att_item[:, [1]]
                entity_emb = torch.cat([only_item_embed, entity_emb[self.n_items:]], dim=0)
                
                user_emb = user_emb * att_user[:, [0]] + self.user_bert_emb * att_user[:, [1]]
            else:
                # n_items x 128
                merged_item_emb = torch.cat([entity_emb[:self.n_items], transformed_bert_emb], dim=-1)
                # print('merged_item_emb', merged_item_emb.shape)
                # n_items x 64
                # output_item_emb = self.output_dropout(self.merge_transform(merged_item_emb))
                output_item_emb = self.item_merge_transform(merged_item_emb)
                # print('output_item_emb', output_item_emb.shape)
                only_item_embed = output_item_emb
                entity_emb = torch.cat([only_item_embed, entity_emb[self.n_items:]], dim=0)
                
                merged_user_emb = torch.cat([user_emb, self.user_bert_emb], dim=-1)
                user_emb = self.user_merge_transform(merged_user_emb)

        if BERT_EMB:
            self.item_bert_emb = self.item_bert_emb.to(user_emb.device)
            if BERT_EMB_NORM:
                # Normalize item embeds
                self.item_bert_emb = self.item_bert_emb /(self.item_bert_emb**2).sum(dim=-1).sqrt()[:, None]
                # print(self.item_bert_emb.shape, 'normalized')
            # n_items x 768
            input_bert_emb = self.item_bert_emb
             # n_items x 64
            # transformed_bert_emb = self.input_dropout(self.input_transform(input_bert_emb))
            # transformed_bert_emb = self.input_transform(input_bert_emb)
            transformed_bert_emb = input_bert_emb # already using pooler to reduce to 64d

            if BERT_EMB_SUM:
                # Pass through linear layers
                transformed_bert_emb = self.merge_transform(transformed_bert_emb)
                # Add to item embeddings
                only_item_embed = transformed_bert_emb + entity_emb[:self.n_items]
                # print('only_item_embed', only_item_embed.shape)
            elif BERT_EMB_ATT_SUM:
                # Compute attention on BERT embeddings
                # n_items x 64 matmul 64 x 64 matmul 64 x n_items
                temp = torch.matmul(transformed_bert_emb, self.comp_att_matrix)
                # print('temp', temp.shape)
                # n_items x 1
                att_value = F.sigmoid(torch.sum(temp*entity_emb[:self.n_items], dim=1)).unsqueeze(dim=-1)
                # print('att_value', att_value.shape)
                # n_items x 1 * n_items x emb_size
                only_item_embed = att_value * transformed_bert_emb + (1-att_value) * entity_emb[:self.n_items]
                # print('only_item_embed', only_item_embed.shape)
            elif BERT_EMB_ATT_SUM_V2:
                # Compute attention on BERT embeddings
                # n_items x 64 matmul 64 x 64 matmul 64 x n_items
                temp = torch.matmul(transformed_bert_emb, self.comp_att_matrix)
                # print('temp', temp.shape)
                # n_items x 1
                att_value = torch.exp(torch.sum(temp*entity_emb[:self.n_items], dim=1)).unsqueeze(dim=-1)
                # print('att_value', att_value.shape)
                # n_items x 1 * n_items x emb_size
                only_item_embed = att_value * transformed_bert_emb + entity_emb[:self.n_items]
                # print('only_item_embed', only_item_embed.shape)
            elif BERT_EMB_MUL:
                # multiply BERT embed with item embed
                only_item_embed = transformed_bert_emb * entity_emb[:self.n_items]
            else:
                # n_items x 128
                merged_item_emb = torch.cat([entity_emb[:self.n_items], transformed_bert_emb], dim=-1)
                # print('merged_item_emb', merged_item_emb.shape)
                # n_items x 128
                # output_item_emb = self.output_dropout(self.merge_transform(merged_item_emb))
                output_item_emb = self.merge_transform(merged_item_emb)
                output_item_emb = self.output_dropout(output_item_emb)
                # print('output_item_emb', output_item_emb.shape)
                # residual
                # only_item_embed = output_item_emb + entity_emb[:self.n_items]
                only_item_embed = output_item_emb

            # item_emb[:self.n_items] = only_item_embed
            entity_emb = torch.cat([only_item_embed, entity_emb[self.n_items:]], dim=0)
        
        # Start processing WL test features
        WL_SUM = check_module_setting('WL_SUM', self.config)
        WL_CONCAT = check_module_setting('WL_CONCAT', self.config)
        WL_RESIDUAL = check_module_setting('WL_RESIDUAL', self.config)
        WL_ATT_SUM = check_module_setting('WL_ATT_SUM', self.config)

        if WL_SUM:
            # n_entities x emb_size
            wl_emb = self.wl_role_emb[self.wl_role_matrix]
            transformed_wl_emb = self.wl_transform_layer(wl_emb)
            entity_emb = entity_emb + wl_emb
            # print('item_emb', item_emb.shape)

        if WL_CONCAT:
            # n_entities x emb_size
            wl_emb = self.wl_role_emb[self.wl_role_matrix]
            # n_entities x 2 emb_size 
            concat_emb = torch.cat([entity_emb, wl_emb], dim=-1)
            # print('concat_emb', concat_emb.shape)
            if WL_RESIDUAL:
                entity_emb = entity_emb + self.wl_concat_dropout(self.wl_concat_layer(concat_emb))
            else:
                entity_emb = self.wl_concat_dropout(self.wl_concat_layer(concat_emb))
            # print('item_emb', item_emb.shape)

        if WL_ATT_SUM:
            # n_entities x emb_size
            wl_emb = self.wl_role_emb[self.wl_role_matrix]
            temp = torch.mm(wl_emb, self.comp_att_matrix)
            # n_entities x 1
            att_value = F.sigmoid(torch.sum(temp*entity_emb, dim=1)).unsqueeze(dim=-1)
            # print('att_value', att_value.shape)
            # n_entities x 1 * n_entities x emb_size
            entity_emb = att_value * wl_emb + (1-att_value) * entity_emb
            # print('item_emb', item_emb.shape)

        return  entity_emb, user_emb


    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor
        # NO_DCORR removes correlation loss from training
        NO_DCORR = check_module_setting('NO_DCORR', self.config)
        if NO_DCORR:
            return mf_loss + emb_loss, mf_loss, emb_loss, cor
        else:
            return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor_loss, cor
