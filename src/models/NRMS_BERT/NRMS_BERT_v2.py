import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict
from models.BERT_modified import BertForEmbeddingExtractionWithPooler
from transformers import BertConfig


class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x, attn_mask=None):

        bz = x.shape[0]
        e = self.att_fc1(x) # (bz, seq_len, 200)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e) # (bz, seq_len, 1)
        
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha) # (bz, 200, 1)
        x = torch.reshape(x, (bz, -1)) # (bz, 400)
        return x

class NRMS(nn.Module):
    def __init__(self, config, tokenized_descriptions):
        super(NRMS, self).__init__()

        self.config = config
        self.tokenized_descriptions = tokenized_descriptions.copy()
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertForEmbeddingExtractionWithPooler(bert_config)
        self.bert_emb_size = bert_config.hidden_size
        self.emb_size = 64
        self.att_pooling = AttentionPooling(self.emb_size, self.emb_size, 0.1)
        # self.item_encoder = ItemEncoder(config, self.bert)
        # self.user_encoder = UserEncoder(config, self.bert)
        initializer = nn.init.xavier_uniform_
        # self.attentioninitializer(torch.empty(768))
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([4]))

    def normalized_batch_vec(self, a):
        return a
        # return a /(a**2).sum(dim=-1).sqrt()[:, None]

    def forward(self, batch=None):
        compute_loss = batch.get('compute_loss', True)
        test_mode = batch.get('test_mode', '')
        if test_mode:
            # Run test with inputs
            if test_mode == 'user_embed':
                # generate user embeds from histories
                users = batch.get('users', None)
                # print(users, users.device)
                device = users.device
                user_histories = batch['user_histories']
                user_history_att_masks = batch.get('user_history_att_masks', None)
                batch_size = len(users)
                seq_len = len(user_histories[0])
                history_ids = user_histories
                # flatten all history ids
                flatten_history_ids = history_ids.view(-1).to(device) 
                # print('flatten_history_ids', flatten_history_ids.shape)
                history_inputs = {
                    'input_ids': self.tokenized_descriptions['input_ids'][flatten_history_ids].to(device),
                    'attention_mask': self.tokenized_descriptions['attention_mask'][flatten_history_ids].to(device),
                    'token_type_ids': self.tokenized_descriptions['token_type_ids'][flatten_history_ids].to(device),
                }
                # print(history_inputs['input_ids'].device)
                # batch_size*seq_len x hidden_size
                history_outputs = self.bert(**history_inputs)
                user_history = history_outputs.view(batch_size, seq_len, -1)
                pooled_output = self.att_pooling(user_history, attn_mask=user_history_att_masks)
                pooled_output = self.normalized_batch_vec(pooled_output)
                # print('pooled_output', pooled_output.shape)
                user_embed = pooled_output#[:, :, None]
                return user_embed
            elif test_mode == 'item_embed':
                pos_items = batch.get('pos_items', None)
                device = pos_items.device
                item_ids = pos_items.view(-1)
                item_inputs = {
                    'input_ids': self.tokenized_descriptions['input_ids'][item_ids].to(device),
                    'attention_mask': self.tokenized_descriptions['attention_mask'][item_ids].to(device),
                    'token_type_ids': self.tokenized_descriptions['token_type_ids'][item_ids].to(device),
                }
                item_outputs = self.bert(**item_inputs)
                item_embed = self.normalized_batch_vec(item_outputs)
                return item_embed

        else:
            users = batch['users']
            # print('user device', users.device)
            device = users.device
            # batch_size
            pos_items = batch.get('pos_items', None)
            # batch_size x 4
            neg_items = batch.get('neg_items', None)
            user_histories = batch['user_histories']
            user_history_att_masks = batch.get('user_history_att_masks', None)
            batch_size = len(users)
            seq_len = len(user_histories[0])
            # print('batch_size {} seq_len {}'.format(batch_size, seq_len))
            # whether it is training with positive samples
            # print('pos_items', pos_items.shape)
            # print('neg_items', neg_items.shape)
            item_ids = torch.cat([pos_items.view(batch_size, 1), neg_items],dim=-1)
            # print('item_ids', item_ids.shape)
            item_ids = item_ids.view(-1)

            targets = torch.LongTensor([1, 0, 0, 0, 0]*batch_size).view(batch_size, -1).to(device)
            flat_targets = torch.LongTensor([0]*batch_size).to(device)
            # print('targets', targets)

            self.tokenized_descriptions['input_ids'] = self.tokenized_descriptions['input_ids'].to(device)
            self.tokenized_descriptions['attention_mask'] = self.tokenized_descriptions['attention_mask'].to(device)
            self.tokenized_descriptions['token_type_ids'] = self.tokenized_descriptions['token_type_ids'].to(device)

            item_inputs = {
                'input_ids': self.tokenized_descriptions['input_ids'][item_ids].to(device),
                'attention_mask': self.tokenized_descriptions['attention_mask'][item_ids].to(device),
                'token_type_ids': self.tokenized_descriptions['token_type_ids'][item_ids].to(device),
            }
            # print(item_inputs)
            
            # Process to align histories
            # for each 
            history_ids = user_histories
            # print('history_ids', history_ids.shape)
            # history_ids = torch.zeros(batch_size, seq_len)
            # for history in user_histories:
            #     history_ids = torch.LongTensor(history)
            # flatten all history ids
            flatten_history_ids = history_ids.view(-1).to(device) 
            # print('flatten_history_ids', flatten_history_ids.shape)
            history_inputs = {
                'input_ids': self.tokenized_descriptions['input_ids'][flatten_history_ids].to(device),
                'attention_mask': self.tokenized_descriptions['attention_mask'][flatten_history_ids].to(device),
                'token_type_ids': self.tokenized_descriptions['token_type_ids'][flatten_history_ids].to(device),
            }
            # print(history_inputs['input_ids'].device)
            # batch_size*seq_len x hidden_size
            history_outputs = self.bert(**history_inputs)
            # print('history_outputs', history_outputs.shape)
            # batch_size*5 x hidden_size
            item_outputs = self.bert(**item_inputs)
            item_outputs = self.normalized_batch_vec(item_outputs)
            # print('item_outputs', item_outputs.shape)
            item_embed = item_outputs.view(batch_size, -1, self.emb_size)
            # print('item_embed', item_embed.shape)
            user_history = history_outputs.view(batch_size, seq_len, -1)
            # print('user_history', user_history.shape)
            pooled_output = self.att_pooling(user_history, attn_mask=user_history_att_masks)
            pooled_output = self.normalized_batch_vec(pooled_output)
            # print('pooled_output', pooled_output.shape)
            user_embed = pooled_output[:, :, None]
            # batch_size x 5 x 1
            score = torch.bmm(item_embed, user_embed).squeeze(-1)
            # print('score', score)
            
            if compute_loss:
                # loss = self.create_bpr_loss(score, targets)
                # loss = self.loss_fn(score.view(-1).float(), targets.view(-1).float())
                loss = self.loss_fn(score, flat_targets)
                # loss = torch.mean((F.sigmoid(score.squeeze(-1)) - targets) ** 2)
                # print(loss)
                # input()
                return loss, score, targets
            else:
                return score

    def create_bpr_loss(self, score, targets):
        """Create loss for BP

        Args:
            score (Tensor): ratings of items batch_size x (#pos + #neg)
            targets (Tensor): 1 for pos sample, 0 for negative
        """
        batch_size = score.shape[0]
        num_samples_per_batch = score.shape[1]
        num_neg_samples = num_samples_per_batch - 1
        # print(score)
        # pos_scores = batch_size*num_neg_samples x 1
        pos_scores = score[:, 0].view(batch_size, -1).repeat(num_neg_samples, 1)
        # print(pos_scores.shape)
        # print(pos_scores)
        # neg_scores = batch_size*num_neg_samples x 1
        neg_scores = score[:, 1:].T.reshape(batch_size*num_neg_samples, 1)
        # print(neg_scores.shape)
        # print(neg_scores)
        # pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        # neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        # input()
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        return mf_loss