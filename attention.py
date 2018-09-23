import torch
import torch.nn as nn
import torch.nn.functional as F

def penalization_term(a):
    aat = torch.bmm(a, a.transpose(1, 2))
    batch_size, N = aat.size()[:2]
    m = aat - torch.eye(N, device='cuda').repeat(batch_size, 1, 1).view(-1, N, N)
    return torch.sum(m**2)**0.5 / batch_size

class StructuredSelfAttention(nn.Module):

    def __init__(self, input_shape, d_a, r):
        """
            Args:
                input_shape : ({int}, {int}, {int}) input shape (C, H, W)
                d_a         : {int} hidden dimension for the dense layer
                r           : {int} attention-hops or attention heads (num classes)
        """
        super(StructuredSelfAttention,self).__init__()

        self.in_h, self.in_w, self.in_c = input_shape
        self.d_a = d_a
        self.r = r

        self.encoded, self.u = self.prepare_encode()
        self.dropout = nn.Dropout(0.7)
        self.W_s1 = nn.Linear(self.u, self.d_a)
        self.W_s2 = nn.Linear(self.d_a, self.r)
        self.W_out = nn.Conv2d(self.u, 1, kernel_size=1)

    def forward(self, x):
        batch_size = x.size()[0]
        in_flatten = self.in_h * self.in_w

        # build and add coords
        encoded = self.encoded.repeat(batch_size, 1, 1)
        encoded = encoded.view(batch_size, -1, in_flatten).cuda()
        h = x.view(batch_size, -1, in_flatten)
        h = torch.cat([h, encoded], dim=1).transpose(1, 2)

        # self attention
        x = F.tanh(self.dropout(self.W_s1(h)))
        a = F.softmax(self.W_s2(x))
        m = torch.bmm(a.transpose(1, 2), h).transpose(1, 2)
        m = self.W_out(m.view(batch_size, self.u, self.r, 1))

        return m.view(-1, self.r), a

    def prepare_encode(self):
        from itertools import product
        import numpy as np

        encoded = np.vstack([np.concatenate((x, y)) \
            for x, y in product(np.eye(self.in_h), np.eye(self.in_w))])
        encoded = np.expand_dims(encoded, axis=0).astype(np.float32)
        encoded = torch.from_numpy(encoded)
        encoded = encoded.transpose(1, 2)
        return encoded, self.in_c + self.in_h + self.in_w
