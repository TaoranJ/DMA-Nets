import datetime
from itertools import chain
from dateutil.relativedelta import relativedelta

import torch
import numpy as np


eps = torch.tensor(1e-9)


def epoch2date(epoch):
    """Epoch -> date.

    Parameters
    ----------
    Epoch : float
        Date in epoch format.

    Returns
    -------
    years: int
        Number of years.
    months: int
        Number of months.
    days: int
        Number of days.

    """

    timedelta = datetime.timedelta(days=epoch)
    diff = relativedelta(timedelta + datetime.datetime(1970, 1, 1),
                         datetime.datetime(1970, 1, 1))
    return diff.years, diff.months, diff.days


def epoch2ymd(epoch, pad_mask):
    """From epoch -> year, month, day

    Parameters
    ----------
    epoch : :class:`torch.Tensor`
        Epoch is a padded sequence, (len, N, 1).
    pad_mask : :class:`torch.Tensor`
        Mask padded positions. A tensor of shape (len, N, 1).

    Returns
    -------
    Epoch in years, months and days format.

    """

    ymd = epoch.flatten().cpu().numpy().tolist()
    ymd = list(chain.from_iterable(map(epoch2date, ymd)))
    ymd = np.array(ymd).reshape(epoch.size(0), epoch.size(1), 3)
    epoch = torch.from_numpy(ymd).float().to(epoch.device)
    # replace 0 with a very small number (1e-9)
    epoch = torch.max(epoch, torch.zeros(1).fill_(1e-9).to(epoch.device))
    return epoch * pad_mask


def z2eps(tensor):
    """Replace zero with eps.

    Parameters
    ----------
    tensor : :class:`torch.Tensor`
        A tensor.

    """

    tensor[tensor == 0] = eps
    return tensor


def reorder_labels(data):
    """Sometimes, there are missing labels in the raw data, e.g., 1, 2, 4 are
    used but 3 is missing. Softmax layer doesn't like it, so we need to reorder
    the labels, i.e., {1 -> 1, 2 -> 2, 4 -> 3}.

    Parameters
    ----------
    data : :class:`collections.defaultdict`
        Dataset: :code:`{<attr_name>: [seq 1, ..., seq n]}`, where each seq is
        a list.

    Returns
    -------
    data : :class:`collections.defaultdict`
        Data with labels reordered.
    n2o : dict
        {new label: old label}.

    """

    labels = sorted(list(set(chain(*chain.from_iterable(  # collect all labels
        [val for val in data.values() if type(val[0][0]) == int])))))
    # n = new labels, o = old labels
    n2o = {new: old for new, old in enumerate(labels, start=1)}
    o2n = {old: new for new, old in enumerate(labels, start=1)}
    for key, seqs in data.items():  # re-assign the new labels to data
        if key[0] == 'm' and 's2s_cut' not in key:  # 'mxxx' refers to the mark
            for idx, seq in enumerate(seqs):
                data[key][idx] = [o2n[old_label] for old_label in seq]
    return data, n2o
