import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, num_heads, num_layers=1):
        super().__init__()
        layers = list()
        for i in range(num_layers):
            layers.append(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
            )
        self.layers = nn.ModuleList(layers)
        
    def forward(self, original_emb, self_attention=True):
        """[summary]

        Args:
            original_emb (Tensor): batch x emb_size*2
            external_emb (Tensor): batch x emb_size*2
        """
        batch_size = original_emb.shape[0]
        if self_attention:
            # batch x 2 x emb_size
            original_emb = original_emb.view(batch_size, 2, -1).permute(1,0,2)
            external_emb = original_emb
        # print('original_emb', original_emb.shape)
        # print('external_emb', external_emb.shape)
        key = original_emb
        value = original_emb
        query = external_emb
        for att_layer in self.layers:
            attn_output, attn_output_weights = att_layer(query, key, value)
            # turn to self attention
            query = attn_output
            key = attn_output
            value = attn_output
        attn_output = attn_output.permute(1,0,2).reshape(batch_size, -1)
        # print('attn_output', attn_output.shape)
        # print('attn_output_weights', attn_output_weights.shape)
        return attn_output

class MultiLayerPerceptron(torch.nn.Module):
    """
    Class to instantiate a Multilayer Perceptron model
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)