import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from layers import MultiHeadAttention


class Encoder(nn.Module):
    """The encoder.

    Parameters
    ----------
    encoder_input_layer : :class:`nn.Module`
        Encoder's input layer.
    encoder_hidden_size : int
        Size of hidden state of the encoder.
    n_heads : bool
        Number of heads used in LTA.
    d_q : int
        Size of head-wise query.
    d_k : int
        Size of head-wise key.
    d_v : int
        Size of head-wise value.
    dropout : float
        Dropout rate.

    """

    def __init__(self, encoder_input_layer, encoder_hidden_size, n_heads, d_q,
                 d_k, d_v, dropout):
        super(Encoder, self).__init__()
        # define input layer
        self.encoder_input_layer = encoder_input_layer
        # define RNN layer
        self.encoder = nn.LSTM(input_size=self.encoder_input_layer.input_size,
                               hidden_size=encoder_hidden_size)
        # define LTA layer
        self.bwk_attn = MultiHeadAttention(
                n_heads, encoder_hidden_size, encoder_hidden_size,
                encoder_hidden_size, d_q, d_k, d_v, dropout,
                query_transform=True)

    def encode(self, ob, pad_mask):
        """Encode observation side.

        Parameters
        ----------
        ob : dict
            Observations: timestamp sequences denoted by ob['t1'],
            document category sequences denoted by ob['m1']
            and their corresponding sequence length ob['len1'].
            Both are padded sequences
            of shape either (max_len, batch, 1) or (batch).
        pad_mask : :class:`torch.Tensor`
            Mask padded positions, a tensor of shape (max_len, batch, 1).

        Returns
        -------
        inputs : :class:`torch.Tensor`
            Encoder's embedded inputs, a tensor of shape
            (max_len, batch, input_size), where input_size = 3 + e_emb_size.
        outputs : :class:`torch.Tensor`
            Encoder's outputs, a tensor of shape
            (max_len, batch, encoder_hidden_size).
        hn : :class:`torch.Tensor`
            Encoder's last hidden state, tensor of shape
            (num_layers, batch, encoder_hidden_size)
        hc : :class:`torch.Tensor`
            Encoder's last cell state, tensor of shape
            (num_layers, batch, encoder_hidden_size).

        """

        inputs = self.encoder_input_layer(ob, pad_mask)
        outputs, (hn, hc) = self.encoder(inputs)
        inputs = pad_packed_sequence(inputs)[0]
        outputs = pad_packed_sequence(outputs)[0]
        return inputs, outputs, hn, hc

    def forward(self, ob, pad_mask=None, attn_mask=None):
        """Forward propagation.

        Parameters
        ----------
        ob : dict
            Observations: timestamp sequences denoted by ob['t1'],
            document category sequences denoted by ob['m1']
            and their corresponding sequence length ob['len1'].
            Both are padded sequences
            of shape either (max_len, batch, 1) or (batch).
        pad_mask : :class:`torch.Tensor`
            Mask padded positions, a tensor of shape (max_len, batch, 1). This
            is for input.
        attn_mask : :class:`torch.Tensor`
            Mask padded positions and links to future, a tensor of shape
            (batch, max_len, max_len). This is for attention calculation.

        Returns
        -------
        inputs : :class:`torch.Tensor`
            Encoder's embedded inputs, a tensor of shape
            (max_len, batch, input_size), where input_size = 3 + e_emb_size.
        states : :class:`torch.Tensor`
            LTA's states, tensor of shape
            (max_len, batch, encoder_hidden_size).
        hn : :class:`torch.Tensor`
            Encoder's last hidden state, tensor of shape
            (num_layers, batch, encoder_hidden_size)
        hc : :class:`torch.Tensor`
            Encoder's last cell state, tensor of shape
            (num_layers, batch, encoder_hidden_size).

        """

        inputs, outputs, hn, hc = self.encode(ob, pad_mask)
        states = self.bwk_attn(outputs, outputs, outputs, attn_mask)
        return inputs, states, hn, hc
