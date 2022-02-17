import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from utils import epoch2ymd


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Parameters
    ----------
    n_heads : int
        Number of heads to use.
    d_Q : int
        Size of query Q.
    d_K : int
        Size of key K.
    d_V : int
        Size of value V.
    d_q : int
        Size of head-wise query.
    d_k : int
        Size of head-wise key.
    d_v : int
        Size of head-wise value.
    dropout : float
        Dropout rate.
    query_transform : bool
        Apply a non-linear ransformation on query Q first.

    """

    def __init__(self, n_heads, d_Q, d_K, d_V, d_q, d_k, d_v, dropout,
                 query_transform=False):
        super().__init__()

        # multi-head configuration
        self.n_heads, self.d_q, self.d_k, self.d_v = n_heads, d_q, d_k, d_v
        # define the optional transformation layer for Q.
        if query_transform:
            self.tran_Q = nn.Sequential(nn.Linear(d_Q, d_Q, bias=False),
                                        nn.Sigmoid())
        self.Q2q = nn.Linear(d_Q, n_heads * d_q, bias=False)
        self.K2k = nn.Linear(d_K, n_heads * d_k, bias=False)
        self.V2v = nn.Linear(d_V, n_heads * d_v, bias=False)
        # score function to use
        self.attn = DotProductAttentionMH(temperature=d_k, dropout=dropout)
        # in case n_heads * d_v != d_V
        if d_V != n_heads * d_v:
            self.v2V = nn.Linear(n_heads * d_v, d_V, bias=False)
        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """Multi-Head Attention forwarding.

        Parameters
        ----------
        Q : :class:`torch.tensor`
           Query, a tensor of shape (len_q, batch, d_Q).
        K : :class:`torch.tensor`
           Key, a tensor of shape (len_k, batch, d_K).
        V : :class:`torch.tensor`
           Value, a tensor of shape (len_v, batch, d_V).
        mask : :class:`torch.tensor`
           Mask padded position or/and illegal attentions, a tensor of shape
           (batch, 1, len_q) or (batch, len_q, len_k).

        Returns
        -------
        Context : :class:`torch.tensor`
            Context, a tensor of shape (len_q, batch, d_V).

        """

        # # of keys and values should be same
        assert(K.size(0) == V.size(0))
        # apply a non-linear transformation on Q first
        if hasattr(self, 'tran_Q'):
            Q = self.tran_Q(Q)
        # Q/K/V -> heads * q/k/v (multi-heads), (batch, heads, len, d_q/k/v)
        Q_ = self.Q2q(Q).view(Q.size(1), Q.size(0), self.n_heads,
                              self.d_q).transpose(1, 2)
        K_ = self.K2k(K).view(K.size(1), K.size(0), self.n_heads,
                              self.d_k).transpose(1, 2)
        V_ = self.V2v(V).view(V.size(1), V.size(0), self.n_heads,
                              self.d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        # head-wise context (batch, len_q, n_heads, d_v)
        context = self.attn(Q_, K_, V_, mask=mask).transpose(1, 2).contiguous()
        # (batch, len_q, n_heads, d_v -> n_heads * d_v)
        context = context.transpose(1, 2).contiguous().view(
                context.size(0), context.size(1), -1)
        if hasattr(self, 'v2V'):
            return self.dropout(self.v2V(context)).transpose(0, 1)
        return self.dropout(context).transpose(0, 1)


class DotProductAttentionMH(nn.Module):
    """Scaled dot product attention layer, multi-head version.

    softmax(QK/scale)V

    Parameters
    ----------
    temperature : float
        Scaling factor of QK.
    dropout : float
        Dropout rate.

    """

    def __init__(self, temperature=1, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """Scaled dot product attention layer forwarding.

        Parameters
        ----------
        Q : :class:`torch.tensor`
           Query Q, a tensor of shape (batch, n_heads, len_q, d_q).
        K : :class:`torch.tensor`
           Key K, a tensor of shape (batch, n_heads, len_k, d_k).
        V : :class:`torch.tensor`
           Value V, a tensor of shape (batch, n_heads, len_v, d_v).
        mask : :class:`torch.tensor`
           Mask padded position and/or links to future, a tensor of shape
           (batch, 1, len_q) or (batch, 1, len_q, len_k).

        Returns
        -------
        context : :class:`torch.tensor`
            Context, a tensor of shape (batch, n_heads, len_q, d_v). Each query
            in len_q is a weighted aggregation of d_v.

        """

        assert(Q.size(3) == K.size(3))  # d_q == d_k for dot product
        # (batch, n_heads, len_q, d_q) @ (batch, n_heads, d_k, len_k)
        attn = torch.matmul(Q / self.temperature, K.transpose(2, 3))
        # -\infty = ignore this position (mask = 0)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # (batch, n_heads, len_q, len_k), softmax on 'key' dimension.
        weights = self.dropout(F.softmax(attn, dim=-1))
        # (batch, n_heads, len_q, len_k) @ (batch, n_heads, len_v, d_v)
        return torch.matmul(weights, V)


class TMPrediction(nn.Module):
    """Prediction layer for event timestamp (t1) and event type (m1).

    Parameters
    ----------
    state_size : int
        Size of the state of the model.
    num_categories : int
        Number of categories for classification task.

    """

    def __init__(self, state_size, num_categories):
        super(TMPrediction, self).__init__()
        self.t_pred = nn.Sequential(nn.Linear(state_size, 1), nn.ReLU())
        self.m_pred = nn.Sequential(nn.Linear(state_size, num_categories),
                                    nn.LogSoftmax(dim=-1))
        # get the name of the target info seq, e.g., (t2, m2, len2)
        self.t, self.m = 't1', 'm1'

    def forward(self, state):
        """Forward prediction.

        Parameters
        ----------
        state : :class:`torch.Tensor`
            Model state, a tensor of shape (max_len, batch, state_size).

        Returns
        -------
        {self.t: timestamp prediction, self.m: category prediction}

        """

        return {self.t: self.t_pred(state), self.m: self.m_pred(state)}


class EventSeqEmb_RNN_YMD(nn.Module):
    """Similar to EventSeqEmb_RNN, but the inter-arrival duration is converted
    to years, months and days. For example, 34 days -> 0 year, 1 month, 4 days.

    Parameters
    ----------
    num_categories : int
        Number of categories of event type.
    e_emb_size : int
        Size of event type embedding.
    t_emb_size : int
        Size of event timestamp embedding.

    """
    def __init__(self, num_categories, e_emb_size, t_emb_size):
        super(EventSeqEmb_RNN_YMD, self).__init__()
        if t_emb_size > 3:
            self.t_embed = nn.Linear(3, t_emb_size)
        else:
            t_emb_size = 3
        # set up embedding layers
        self.m_embed = nn.Embedding(num_categories, e_emb_size, padding_idx=0)
        # get the name of the target info seq, e.g., (t2, m2, len2)
        self.t, self.m = 't1', 'm1'
        self.len = 'len1'
        # for downstream tasks
        self.input_size = t_emb_size + e_emb_size

    def forward(self, data, pad_mask):
        """Forward propagation.

        Parameters
        ---------
        data : dict
            Given a batch of event sequences
            [[e1, e2, ...], [e1, e2, ...], ...], data[self.t] refers to the
            corresponding event timestamp info, a tensor of shape
            (max_len, batch, 1) and data[self.m] is the corresponding event
            type info, a tensor of shape (max_len, batch, 1). Both of these two
            tensors are already padded in dataloader. data[self.len] records
            the length for each event sequence, a tensor of shape (1, batch).
        pad_mask : :class:`torch.tensor`
            Mask padded positions in the inputs, a tensor of shape
            (len, batch, 1).

        """

        t = epoch2ymd(data[self.t], pad_mask)
        t = self.t_embed(t) if hasattr(self, 't_embed') else t
        m = self.m_embed(data[self.m].squeeze(-1))
        inputs = torch.cat([t, m], dim=-1)
        if inputs.size(0) > 1:  # if the length of seq is > 1
            return pack_padded_sequence(inputs,
                                        data[self.len].squeeze(0).cpu(),
                                        enforce_sorted=False)
        return inputs
