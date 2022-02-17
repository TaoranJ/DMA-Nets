import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

from layers import MultiHeadAttention


class Decoder(nn.Module):
    """The decoder.

    Parameters
    ----------
    decoder_input_layer : :class:`nn.Module`
        Decoder's input layer.
    decoder_hidden_size : int
        Size of hidden state of the decoder.
    n_heads : bool
        Number of heads used in LTA.
    d_q : int
        Size of head-wise query.
    d_k : int
        Size of head-wise key.
    d_v : int
        Size of head-wise value.
    m1 : int
        Number of heads used in GMTA, context 1.
    m2 : int
        Number of heads used in GMTA, context 2.
    pred : :class:`nn.Module`
        Prediction layer.
    dropout : float
        Dropout rate.

    """

    def __init__(self, decoder_input_layer, decoder_hidden_size, n_heads, d_q,
                 d_k, d_v, m1, m2, pred, dropout):
        super(Decoder, self).__init__()
        # define input layer
        self.decoder_input_layer = decoder_input_layer
        # define RNN layer
        self.decoder = nn.LSTM(input_size=self.decoder_input_layer.input_size,
                               hidden_size=decoder_hidden_size)
        # define LTA layer
        self.bwk_attn = MultiHeadAttention(
                n_heads, decoder_hidden_size, decoder_hidden_size,
                decoder_hidden_size, d_q, d_k, d_v, dropout,
                query_transform=True)
        # define GMTA layer
        emb_size = self.decoder_input_layer.input_size
        self.static1 = MultiHeadAttention(  # c^{e1}
                m1, decoder_hidden_size, decoder_hidden_size,
                decoder_hidden_size, decoder_hidden_size // m1,
                decoder_hidden_size // m1, decoder_hidden_size // m1, dropout)
        self.static2 = MultiHeadAttention(  # c^{e2}
                m2, emb_size, emb_size, decoder_hidden_size, emb_size // m2,
                emb_size // m2, decoder_hidden_size, dropout)
        self.dynamic1 = MultiHeadAttention(  # c^{d1}
                m1, decoder_hidden_size, decoder_hidden_size,
                decoder_hidden_size, decoder_hidden_size // m1,
                decoder_hidden_size // m1, decoder_hidden_size // m1, dropout)
        self.dynamic2 = MultiHeadAttention(  # c^{d2}
                m2, emb_size, emb_size, decoder_hidden_size, emb_size // m2,
                emb_size // m2, decoder_hidden_size, dropout)
        self.blend = nn.Sequential(  # blend
                nn.Linear(decoder_hidden_size * 4, decoder_hidden_size),
                nn.GELU(), nn.Dropout(dropout))
        # setup prediction layer
        self.pred = pred

    def decode(self, tgt, hn, hc, pad_mask):
        """Decode.

        Parameters
        ----------
        tgt : :class:`torch.Tensor`
            tgt contains the input at current step, a tensor of shape
            (1/max_len, batch, 1).
        hn : :class:`torch.Tensor`
            Decoder's last hidden state, tensor of shape
            (1, batch, decoder_hidden_size).
        hc : :class:`torch.Tensor`
            Decoder's last cell state, tensor of shape
            (1, batch, decoder_hidden_size).
        pad_mask : :class:`torch.Tensor`
            Mask padded positions of current input, a tensor of shape
            (1/max_len, batch, 1).

        Returns
        -------
        input : :class:`torch.Tensor`
            Embedded input, a tensor of shape (1/max_Len, batch, input_size).
        output : :class:`torch.Tensor`
            Decoder's output at current step, a tensor of shape
            (1/max_len, batch, decoder_hidden_size)
        hn : :class:`torch.Tensor`
            Decoder's hidden state at current step, a tensor of shape
            (num_layers, batch, decoder_hidden_size)
        hc : :class:`torch.Tensor`
            Decoder's cell state at current step, a tensor of shape
            (num_layers, batch, decoder_hidden_size)

        """

        input = self.decoder_input_layer(tgt, pad_mask)
        output, (hn, hc) = self.decoder(input, (hn, hc))
        if 'PackedSequence' in str(type(output)):
            output = pad_packed_sequence(output)[0]
        if 'PackedSequence' in str(type(input)):
            input = pad_packed_sequence(input)[0]
        return input, output, hn, hc

    def forward(self, tgt, hn, hc, dec_inputs, dec_outputs, dec_states,
                enc_inputs, enc_states, enc_pad_mask=None, dec_pad_mask=None,
                dec_attn_mask=None, generator=True):
        """Forward propagation.

        Parameters
        ----------
        tgt : :class:`torch.Tensor`
            tgt contains the input at current step, a tensor of shape
            (1/max_len, batch, 1).
        hn : :class:`torch.Tensor`
            Decoder's last hidden state, tensor of shape
            (1, batch, decoder_hidden_size).
        hc : :class:`torch.Tensor`
            Decoder's last cell state, tensor of shape
            (1, batch, decoder_hidden_size).
        dec_inputs : :class:`torch.Tensor`
            Decoder's all embedded inputs up till current step, a tensor of
            shape (max_len, batch, input_size).
        dec_outputs : :class:`torch.Tensor`
            Decoder's all previous outputs up till current step, a tensor of
            shape (max_len, batch, decoder_hidden_size).
        dec_states : :class:`torch.Tensor`
            Decoder's all previous states up till current step, a tensor of
            shape (max_len, batch, decoder_state_size).
        enc_inputs : list
            Encoder's all inputs, a tensor of shape
            (max_len, batch, input_size).
        enc_states : :class:`torch.Tensor`
            Encoder's all states, a tensor of shape
            (max_len, batch, encoder_state_size).
        enc_pad_mask : :class:`torch.Tensor`
            Mask padded positions of inputs to encoder, a tensor of shape
            (max_len, batch, 1).
        dec_pad_mask : :class:`torch.Tensor`
            Mask padded positions of previous inputs (including current one) to
            decoder, a tensor of shape (max_len, batch, 1). This is for input.
        dec_attn_mask : :class:`torch.Tensor`
            Mask illegal positions (pad/future) to attend, a tensor of shape
            (batch, max_len, max_len). This is for attention calculation.
        generator : bool
            Ture if generator mode is used.

        """

        # run decoder
        if generator:
            dec_input, dec_output, hn, hc = self.decode(
                    tgt, hn, hc, dec_pad_mask[-1].unsqueeze(0))
            dec_outputs = torch.cat(dec_outputs + [dec_output], dim=0)
            # LTA: Q = decoder_output, K, V = decoder_outputs, self included
            dec_state = self.bwk_attn(dec_output, dec_outputs, dec_outputs,
                                      mask=dec_pad_mask.permute(1, 2, 0))
        else:
            dec_input, dec_output, hn, hc = self.decode(tgt, hn, hc,
                                                        dec_pad_mask)
            dec_outputs = dec_output
            # LTA: Q = decoder_output, K, V = decoder_outputs, self included
            dec_state = self.bwk_attn(dec_output, dec_outputs, dec_outputs,
                                      mask=dec_attn_mask)
        # GMTA: Q = decoder_state, K = V = encoder_states
        c_e1 = self.static1(dec_state, enc_states, enc_states,
                            mask=enc_pad_mask.permute(1, 2, 0))
        # GMTA: Q = decoder_input, K = encoder_inputs, V = encoder_states
        c_e2 = self.static2(dec_input, enc_inputs, enc_states,
                            mask=enc_pad_mask.permute(1, 2, 0))
        if generator:
            if dec_states:
                dec_states = torch.cat(dec_states, dim=0)
                dec_inputs = torch.cat(dec_inputs, dim=0)
                # GMTA: Q = decoder_state, K = V = decoder_states
                c_d1 = self.dynamic1(dec_state, dec_states, dec_states,
                                     dec_pad_mask[:-1].permute(1, 2, 0))
                # GMTA: Q = decoder_input, K = decoder_inputs,
                # V = decoder_states
                c_d2 = self.dynamic2(dec_input, dec_inputs, dec_states,
                                     dec_pad_mask[:-1].permute(1, 2, 0))
            else:
                c_d1 = torch.zeros_like(dec_state)
                c_d2 = torch.zeros_like(dec_state)
        else:
            # GMTA: Q = decoder_state, K = V = decoder_states
            c_d1 = self.dynamic1(dec_state, dec_state, dec_state,
                                 dec_attn_mask)
            # GMTA: Q = decoder_input, K = decoder_inputs,
            # V = decoder_states
            c_d2 = self.dynamic2(dec_input, dec_input, dec_state,
                                 dec_attn_mask)
        # sum 2, sum 2, dec_state, hn
        if generator:
            out = self.blend(torch.cat([c_e1 + c_e2, c_d1 + c_d2, dec_state,
                                        hn], dim=2))
        else:
            out = self.blend(torch.cat([c_e1 + c_e2, c_d1 + c_d2, dec_state,
                                        dec_outputs], dim=2))
        pred = self.pred(out)
        return pred, hn, hc, dec_state, dec_input
