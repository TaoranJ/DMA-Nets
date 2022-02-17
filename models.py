import torch
import torch.nn as nn

from args import args
from encoders import Encoder
from decoders import Decoder
from layers import TMPrediction, EventSeqEmb_RNN_YMD


def backward_only_mask(max_len):
    """Mask links to future.

    Parameters
    ----------
    max_len : int
        Max length of in the padded sequences.

    Returns
    -------
    A tensor of shape (1, max_len, max_len).

    """

    return torch.tril(torch.ones(max_len, max_len,
                                 device=args.device)).unsqueeze(0).bool()


class DMANets(nn.Module):
    """DMA-Nets.

    Parameters
    ----------
    t_emb_size : int
        Size of event timestamp embedding.
    num_categories : int
        Number of categories of document category.
    e_embed_size : int
        Size of document category embedding.
    hidden_size : int
        Size of the hidden/cell state of LSTMs.
    n_heads : int
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
    dropout : float
        Dropout rate.

    """

    def __init__(self, t_emb_size, num_categories, e_emb_size, hidden_size,
                 n_heads, d_q, d_k, d_v, m1, m2, dropout):
        super(DMANets, self).__init__()
        # define encoder
        self.encoder_input_layer = EventSeqEmb_RNN_YMD(
                num_categories=num_categories, e_emb_size=e_emb_size,
                t_emb_size=t_emb_size)
        self.encoder = Encoder(
                encoder_input_layer=self.encoder_input_layer,
                encoder_hidden_size=hidden_size, n_heads=n_heads, d_q=d_q,
                d_k=d_k, d_v=d_v, dropout=dropout)
        # define decoder
        self.decoder_input_layer = EventSeqEmb_RNN_YMD(
                num_categories=num_categories, e_emb_size=e_emb_size,
                t_emb_size=t_emb_size)
        self.pred = TMPrediction(state_size=hidden_size,
                                 num_categories=num_categories)
        self.decoder = Decoder(decoder_input_layer=self.decoder_input_layer,
                               decoder_hidden_size=hidden_size,
                               n_heads=n_heads, d_q=d_q, d_k=d_k, d_v=d_v,
                               m1=m1, m2=m2, pred=self.pred, dropout=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, ob, gt, t_type):
        """Generate mask based on the input and pre-defined illegal positions.

        Parameters
        ----------
        ob : dict
            Observations. ob['t1'/'m1'] are padded attribute sequences of shape
            (max_len, batch, 1). ob['len1'] is the corresponding sequence
            length, a tensor of shape (batch).
        gt : dict
            Ground truth. gt['t1'] is the ground truth for patent sequence,
            a tensor of shape (max_len, batch, 1). gt['sos'] is the SOS for
            decoders.

        Returns
        -------
        ob_pad_mask : :class:`torch.tensor`
            A boolean tensor of shape (ob_max_len, batch, 1), where
            0 -> this position is a padding in ob.
        ob_attn_mask : :class:`torch.tensor`
            A boolean tensor of shape (batch, ob_max_len, ob_max_len), where
            0 -> this position is a illegal position (pad/future) in ob.
        gt_pad_mask : :class:`torch.tensor`
            A boolean tensor of shape (gt_max_len, batch, 1), where
            0 -> this position is a padding in gt.
        gt_attn_mask : :class:`torch.tensor`
            A boolean tensor of shape (batch, gt_max_len, gt_max_len), where
            0 -> this position is a illegal position (pad/future) in gt.

        """

        ob_pad_mask = ob['m1'] != 0
        ob_attn_mask = ob_pad_mask.permute(1, 2, 0) \
            & backward_only_mask(ob['m1'].size(0))
        gt_pad_mask = torch.cat([gt['sos'][t_type]['m1'], gt['m1'][:-1]],
                                dim=0) != 0
        gt_attn_mask = gt_pad_mask.permute(1, 2, 0) \
            & backward_only_mask(gt['m1'].size(0))
        return ob_pad_mask, ob_attn_mask, gt_pad_mask, gt_attn_mask

    def forward(self, ob, gt):
        """Forward propagation.

        Parameters
        ----------
        ob : dict
            Observations. ob['t1'/'m1'] are padded attribute sequences of shape
            (max_len, batch, 1). ob['len1'] is the corresponding sequence
            length, a tensor of shape (batch).
        gt : dict
            Ground truth. gt['t1'] is the ground truth for patent sequence,
            a tensor of shape (max_len, batch, 1). gt['sos'] is the SOS for
            decoders.

        """

        t_type = 'tau'
        ob_pad_mask, ob_attn_mask, gt_pad_mask, gt_attn_mask = \
            self.generate_mask(ob, gt, t_type)
        # encode observations
        encoder_inputs, encoder_states, hn, hc = self.encoder(
                ob, pad_mask=ob_pad_mask, attn_mask=ob_attn_mask)
        # make predictions
        decoder_inputs, decoder_outputs, decoder_states = [], [], []
        pred = {'t1': gt['sos'][t_type]['t1'], 'm1': gt['sos'][t_type]['m1']}
        preds = {'t1': [], 'm1': []}
        for ix, _ in enumerate(gt['t1']):
            pred, hn, hc, decoder_state, decoder_input = self.decoder(
                    pred, hn, hc, decoder_inputs, decoder_outputs,
                    decoder_states, encoder_inputs, encoder_states,
                    ob_pad_mask, gt_pad_mask[:ix + 1], None, True)
            decoder_inputs.append(decoder_input)  # embedded inputs
            decoder_outputs.append(hn[-1].unsqueeze(0))  # outputs
            decoder_states.append(decoder_state)  # states
            preds['t1'].append(pred['t1'])  # predicts
            if self.training:  # only need logsofmax for training
                preds['m1'].append(pred['m1'])
            pred['m1'] = pred['m1'].topk(1, dim=-1)[1]
            # collect prediction results for evaluation
            if not self.training:
                preds['m1'].append(pred['m1'])
            # next raw input
            pred = {k: v.detach() for k, v in pred.items()}
        return {k: torch.cat(v) for k, v in preds.items()}
