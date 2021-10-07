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
        ):
        super(EmbeddingNet, self).__init__()
        self.image_nn = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(image_dim, output_dim),
            # nn.SiLU(inplace=True),
        )
        self.text_nn = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(text_dim, output_dim),
            # nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(dropout, inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(output_dim, 4 * output_dim),
            nn.BatchNorm1d(4 * output_dim),
            # nn.LayerNorm((4 * output_dim)),
            # nn.SiLU(inplace=True),
            Mish(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(4 * output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            # nn.LayerNorm((output_dim)),
            # nn.SiLU(inplace=True),
        )

        self.classifier2 = nn.Sequential(
            # nn.Dropout(dropout, inplace=True),
            # nn.Linear(1024, 512),
            nn.Linear(output_dim, 4 * output_dim),
            nn.BatchNorm1d(4 * output_dim),
            # nn.LayerNorm((4 * output_dim)),
            # nn.SiLU(inplace=True),
            Mish(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(4 * output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            # nn.LayerNorm((output_dim)),
            # nn.SiLU(inplace=True),
        )
        
    def forward(self, embedding):
        image_embedding, text_embedding = embedding
        image_embedding = self.image_nn(image_embedding)
        # text_embedding = self.text_nn(text_embedding)
        embedding = torch.cat(
            (
                image_embedding,
                # text_embedding,
            ),
            dim=-1
        )
        embedding = embedding + self.classifier(embedding)
        embedding = embedding + self.classifier2(embedding)

        return embedding


class Block(nn.Module):

    def __init__(self, fc_dim, dropout = 0.5):
        super(Block, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fc_dim, 4 * fc_dim),
        self.bn1 = nn.BatchNorm1d(4 * fc_dim),
        self.activation = Mish(inplace=True)
        self.fc2 = nn.Linear(4 * fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        self._init_params()

        self.classifier = nn.Sequential(
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