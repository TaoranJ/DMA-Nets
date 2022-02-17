import torch.optim as optimizer

from models import DMANets
from data import dataloader_s2s_rnn


def init(args):
    """Initialize data, model and optimizer.

    Parameters
    ----------
    args : dict
        Configurations.

    Returns
    -------
    train : :class:`torch.utils.data.DataLoader`
        Training set.
    valid : :class:`torch.utils.data.DataLoader`
        Validation set.
    test : :class:`torch.utils.data.DataLoader`
        Test set.
    model : :class:`torch.nn.Modeule`
        DMA-Nets.
    optim : :class:`torch.optim`
        Optimizer.

    """

    # initialize data loader
    train, valid, test = dataloader_s2s_rnn(args)
    # initialize model
    model = DMANets(
            t_emb_size=args.t_emb_size, num_categories=args.num_categories,
            e_emb_size=args.e_emb_size, hidden_size=args.hidden_size,
            n_heads=args.n_heads, d_q=args.dq, d_k=args.dk, d_v=args.dv,
            m1=args.m1, m2=args.m2, dropout=args.dropout).to(args.device)
    optim = optimizer.AdamW(model.parameters(), eps=1e-9, lr=args.lr,
                            weight_decay=args.weight_decay)
    return train, valid, test, model, optim
