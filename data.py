import os
import csv
import math
import configparser
from operator import itemgetter
from collections import defaultdict

import numpy as np
from torch import tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import z2eps, reorder_labels

data_config = configparser.ConfigParser()


# =============================================================================
# ============================= Handle Raw Data ===============================
# =============================================================================
def load_dataset(args):
    """Return training/validation/test dataset.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    tr_set : dict
        Training dataset, organized as {<attr>: [seq 1, ..., seq n]}, where
        each seq is a list.
    val_set : list
        Validation dataset, organized as {<attr>: [seq 1, ..., seq n]}, where
        each seq is a list.
    te_set : list
        Test dataset, organized as {<attr>: [seq 1, ..., seq n]}, where each
        seq is a list.

    """

    # read data file
    data = load_csv(args.config)
    # reorder labels and obtain num of categories in the dataset
    data, args.n2o_labels = reorder_labels(data)
    args.num_categories = len(args.n2o_labels) + 1  # 0 = <PAD>
    # split training, validation, and test set
    tr_idx, val_idx, te_idx = split(len(data['t1']), tr_ratio=0.7,
                                    val_ratio=0.1)
    tr_set = {k: v[0][0] if '_s2s_cut' in k else itemgetter(*tr_idx)(v)
              for k, v in data.items()}
    val_set = {k: v[0][0] if '_s2s_cut' in k else itemgetter(*val_idx)(v)
               for k, v in data.items()}
    te_set = {k: v[0][0] if '_s2s_cut' in k else itemgetter(*te_idx)(v)
              for k, v in data.items()}
    return tr_set, val_set, te_set


def load_csv(data_path):
    """Load data files.

    Parameters
    ----------
    data_path : str
        Path to data folder. Both data files and configuration file are
        expected in in data_path.

    Returns
    -------
    data : :class:`collections.defaultdict`
        Raw data, organized as a dict:
        {
            <attr>: [seq 1, ..., seq n],
            <attr_s2s_cut>: [[True/False]],  # cut this seq for seq2seq model
        }.

    """

    data_config.read(data_path)
    data_root, data = os.path.dirname(data_path), defaultdict(list)
    for attr in data_config.sections():
        config = parse_section(data_config[attr])
        ipath = os.path.join(data_root, config['ipath'])
        data[attr + '_s2s_cut'].append([config['s2s_cut']])
        for seq in csv.reader(open(ipath, 'r')):
            data[attr].append(list(map(
                config['d_type'], seq[:config['max_len']])))
    return data


def parse_section(attr):
    """Parse attribute configurations.

    Parameters
    ----------
    attr : :class:`configparser.SectionProxy`
        Configurations of the target attribute.

    Returns
    -------
    config : dict
        'ipath': path to the target data file.
        'd_type': data type. Either float or int.
        'max_len': maximum acceptable length.
        's2s_cut': cut sequences for the seq2seq structure if True.

    """

    return {'ipath': attr['ipath'],
            'd_type': float if attr['d_type'] == 'float' else int,
            'max_len': attr.getint('max_len'),
            's2s_cut': attr.getboolean('s2s_cut')}


def split(total_num, tr_ratio, val_ratio):
    """Select the training, validation, and test subset.

    Parameters
    ----------
    total_num : int
        Total number of samples in the dataset.
    tr_ratio : float
        Percentage of dataset used for training.
    val_ratio : float
        Percentage of dataset used for validation.

    Returns
    -------
    tr_idx : list
        Index of samples selected as training set.
    val_idx : list
        Index of samples selected as validation set.
    te_idx : list
        Index of samples selected as test set.

    """

    if tr_ratio + val_ratio > 1:
        raise ValueError('tr_ratio + val_ratio > 1? Dark magic is not allowed '
                         'here!')
    np.random.seed(42)
    idx = np.arange(total_num)
    np.random.shuffle(idx)
    tr_idx = int(total_num * tr_ratio)
    val_idx = int(total_num * val_ratio)
    return idx[:tr_idx].tolist(), idx[tr_idx: tr_idx + val_idx].tolist(), \
        idx[tr_idx + val_idx:].tolist()


# =============================================================================
# ============================= Define Datasets ===============================
# =============================================================================
class RatioDataset(Dataset):
    """Given a sequence, ratio * len = observations, the rest are ground truth.

    Parameters
    ----------
    data : :class:`collections.defaultdict`
        Data, organized as a dict:
        {
            <attr>: [seq 1, ..., seq n],
            <attr_s2s_cut>: [[True/False]],  # applied ratio split on this seq
        }.
    ob_ratio : float
        Ratio of sequence used as observations.
    use_interval : bool
        Use inter-arrival time.

    """
    def __init__(self, data, ob_ratio):
        super(RatioDataset, self).__init__()
        self.data = data
        self.ob_ratio = ob_ratio
        self.len = len(data['t1'])

    def __len__(self):
        return self.len

    def __getitem__(self, ix):
        """Split the ix-th sequence to observations and ground truth.

        Parameters
        ----------
        ix : int
            Index.

        Returns
        -------
        ob : dict
            One observation sequence {<attr>: seq}.
        gt : dict
            One ground truth sequence {<attr>: seq}.

        """

        ob, gt = {}, {}
        left_min, right_min = 3, 2
        for attr, seqs in self.data.items():
            if '_s2s_cut' in attr:  # not the attribute seq
                continue
            tgt_seq, tgt_len = seqs[ix], len(seqs[ix])
            if self.data[attr + '_s2s_cut']:  # need to cut for seq2seq model
                assert tgt_len >= left_min + right_min
                ob_len = math.ceil(self.ob_ratio * tgt_len)
                if ob_len < left_min:
                    ob_len = left_min
                if ob_len > tgt_len - right_min:
                    ob_len = tgt_len - right_min
                ob[attr], gt[attr] = tgt_seq[:ob_len], tgt_seq[ob_len:]
            else:  # no need to cut
                ob[attr], gt[attr] = tgt_seq, []
        return ob, gt


# =============================================================================
# ============================ Define DataLoader ==============================
# =============================================================================
def dataloader_s2s_rnn(args):
    """The dataloader for classic sequence-to-sequence structure.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    tr_set : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader for training set.
    val_set : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader for validation set.
    te_set : :class:`torch.utils.data.dataloader.DataLoader`
        Dataloader for test set.

    """

    tr_set, val_set, te_set = load_dataset(args)
    tr_set = RatioDataset(tr_set, args.ob_ratio)
    val_set = RatioDataset(val_set, args.ob_ratio)
    te_set = RatioDataset(te_set, args.ob_ratio)
    tr_set = DataLoader(tr_set, batch_size=args.batch_size,
                        collate_fn=collate_fn_s2s, shuffle=True)
    val_set = DataLoader(val_set, batch_size=args.batch_size,
                         collate_fn=collate_fn_s2s, shuffle=True)
    te_set = DataLoader(te_set, batch_size=args.batch_size,
                        collate_fn=collate_fn_s2s, shuffle=True)
    return tr_set, val_set, te_set


def collate_fn_s2s(insts):
    """Get a minibatch for seqs-to-seq model.

    Parameters
    ----------
    insts : list
        A minibatch [(ob, gt), (ob, gt), ...], where both ob and gt has a look
        like {<attr>: seq}.

    Returns
    -------
    ob : dict
        Observation minibatch,
        {'tau'/'t': {<attr>: tensor, <len> + source_id}}. tau=interval.
    gt : dict
        Ground truth minibatch,
        {'tau'/'t': {<attr>: tensor, <len> + source_id}}. tau=interval.

    """

    ob, gt = minibatch_padding(*list(zip(*insts)))
    return ob, gt


def minibatch_padding(raw_ob, raw_gt):
    """Pad observations and ground truth tensor.

    Parameters
    ----------
    raw_ob : list
        Observations.
    raw_gt : tuple
        Ground truth.

    Returns
    -------
    ob : dict
        Observation minibatch,
        {'tau'/'t': {<attr>: tensor, <len> + source_id}}. tau=interval.
    gt : dict
        Ground truth minibatch,
        {'tau'/'t': {<attr>: tensor, <len> + source_id}}. tau=interval.

    """

    ob, gt, attrs = defaultdict(dict), defaultdict(dict), raw_ob[0].keys()
    for attr in attrs:
        source_id = attr[-1]
        # observations: ob['tau'], ob['t'], tau=interval
        ob['tau'][attr] = pad_sequence(
                [z2eps(tensor(o[attr])[1:] - tensor(o[attr])[:-1])  # R
                 if attr[0] == 't' else tensor(o[attr][1:])  # N
                 for o in raw_ob]).unsqueeze(-1)
        ob['t'][attr] = pad_sequence([tensor(o[attr])
                                      for o in raw_ob]).unsqueeze(-1)
        ob['tau']['len' + source_id] = LongTensor([len(o[attr]) - 1
                                                   for o in raw_ob])
        ob['t']['len' + source_id] = LongTensor([len(o[attr])
                                                 for o in raw_ob])
        # ground truth: gt['tau'], gt['t']
        if raw_gt[0][attr]:
            gt['tau'][attr] = pad_sequence(
                    [z2eps(
                        tensor(g[attr]) - tensor([o[attr][-1]] + g[attr][:-1]))
                     if attr[0] == 't' else tensor(g[attr])
                     for o, g in zip(raw_ob, raw_gt)]).unsqueeze(-1)
            gt['t'][attr] = pad_sequence([tensor(g[attr])
                                          for g in raw_gt]).unsqueeze(-1)
            gt['tau']['len' + source_id] = LongTensor([len(g[attr])
                                                       for g in raw_gt])
            gt['t']['len' + source_id] = LongTensor([len(g[attr])
                                                     for g in raw_gt])
    return ob, gt


# =============================================================================
# ============================= Handle Minibatch ==============================
# =============================================================================
def unpacking_s2s(batch, args):
    """Use ob as input, gt[0] as the SOS, and gt[1:] as the ground truth.

    Parameters
    ----------
    batch : tuple
        Minibatch (ob, gt). Ob and gt share similar structure like
        {'tau': {<attr_name>: tensor}}.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    ob : dict
        Observations: {<attr_name>: tensor, 'len1': tensor}.
    tgt : dict
        Ground truth: {<attr_name>: tensor}. If inter-arrival period is used,
        then tgt also has another key-value pair:
        {'sos': {<attr_name>: gt_t[<attr_name>][0]}}.

    """

    ob, gt = batch
    device = args.device
    ob = {key: seq.to(args.device) for key, seq in ob['tau'].items()}
    tgt = {key: seq[1:].to(args.device) for key, seq in gt['tau'].items()
           if key != 'len1'}
    tgt['len1'] = gt['tau']['len1'] - 1
    # only care attributes to predict, that is 't1' and 'm1' here.
    sos = {'tau': {key: seq[0].unsqueeze(0).to(device)
                   for key, seq in gt['tau'].items()
                   if key[-1] == '1' and key[0] in ['t', 'm']},
           't': {key: seq[0].unsqueeze(0).to(device)
                 for key, seq in gt['t'].items()
                 if key[-1] == '1' and key[0] in ['t', 'm']}}
    tgt['sos'] = sos
    return ob, tgt
