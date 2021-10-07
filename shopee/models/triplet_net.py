import torch
import torch.nn as nn

# source:
# https://github.com/adambielski/siamese-triplet/blob/master/networks.py
# we add freezing layers utilities (WARNING: only working for top levels layers, not children)
class TripletNet(nn.Module):
    def __init__(self, embedding_net, layers_to_train=None):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        # self.layers_to_train = layers_to_train
        # if self.layers_to_train is not None:
        #     for name, module in self.embedding_net.named_children():
        #         if name not in self.layers_to_train:
        #             for param in module.parameters():
        #                 param.requires_grad = False

    def forward(self, triplet):
        return (
            self.embedding_net(x).squeeze()
            for x in triplet
        )
#         output1 = self.embedding_net(x1).squeeze()
#         output2 = self.embedding_net(x2).squeeze()
#         output3 = self.embedding_net(x3).squeeze()
#         return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    # def train(self, mode: bool = True):
    #     r"""Sets the module in training mode.

    #     This has any effect only on certain modules. See documentations of
    #     particular modules for details of their behaviors in training/evaluation
    #     mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #     etc.

    #     Args:
    #         mode (bool): whether to set training mode (``True``) or evaluation
    #                     mode (``False``). Default: ``True``.

    #     Returns:
    #         Module: self
    #     """
    #     self.training = mode

    #     if mode and self.layers_to_train is not None:
    #         for name, module in self.embedding_net.named_children():
    #             if name in self.layers_to_train:
    #                 module.train(mode)
    #         return self

    #     for module in self.children():
    #         module.train(mode)
    #     return self
