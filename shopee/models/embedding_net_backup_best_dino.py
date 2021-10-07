import torch
import torch.nn as nn

from .activations import Mish

class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            image_dim=1792,
            text_dim=512,
            output_dim=512,
            dropout=0.5,
            n_blocks=5,
            expansion=2,
        ):
        super(EmbeddingNet, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.expansion = expansion
        self.image_fc = nn.Linear(image_dim, output_dim)
        nn.init.xavier_normal_(self.image_fc.weight)
        nn.init.constant_(self.image_fc.bias, 0)
        # self.output_fc = nn.Linear(output_dim, output_dim)
        # nn.init.xavier_normal_(self.output_fc.weight)
        # nn.init.constant_(self.output_fc.bias, 0)
        # self.text_fc = nn.Linear(text_dim, output_dim)

        blocks = [Block(output_dim, dropout, expansion)] * n_blocks
        self.classifier = nn.Sequential(
            *blocks, 
            # self.output_fc
            )

        
    def forward(self, embedding):
        image_embedding, text_embedding = embedding
        image_embedding = self.image_fc(image_embedding)
        # text_embedding = self.text_nn(text_embedding)
        embedding = torch.cat(
            (
                image_embedding,
                # text_embedding,
            ),
            dim=-1
        )
        embedding = self.classifier(embedding)
        
        return embedding

    def to_string(self):
        return str(dict(
            output_dim = self.output_dim,
            dropout = self.dropout,
            n_blocks = self.n_blocks,
            expansion = self.expansion,
        ))


class Block(nn.Module):

    def __init__(self, fc_dim, dropout = 0.5, expansion = 2):
        super(Block, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc_dim, expansion * fc_dim)
        self.bn1 = nn.BatchNorm1d(expansion * fc_dim)
        self.activation = Mish(inplace=True)
        self.fc2 = nn.Linear(expansion * fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self._init_params()

        self.classifier = nn.Sequential(
            # self.dropout,
            self.fc1,
            self.bn1,
            self.activation,
            self.dropout,
            self.fc2,
            self.bn2,
        )


    def _init_params(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x):
        return x + self.classifier(x)