import torch
from tqdm import tqdm
import torch.nn.functional as F


# =============================================================================
# =============================== Train model =================================
# =============================================================================
def train(model, optim, train_set, valid_set, unzip, args):
    """Training procedure.

    Parameters
    ----------
    model
        The model.
    optim
        The optimizer.
    train_set : :class:`torch.utils.data.DataLoader`
        Training set.
    valid_set : :class:`torch.utils.data.DataLoader`
        Validation set.
    unzip : function
        A function to unpack the minibatch properly.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    model
        Trained model.

    """

    epoch_loss = 0
    for epoch in range(1, args.epochs + 1):
        for batch in tqdm(train_set):
            loss = train_step(model, optim, batch, unzip, args)
            epoch_loss += loss
        res = evaluate(model, valid_set, unzip, args)
        print('[Epochs: {:02d}/{:02d}], Loss: {:.6f}, MAE: {:.4f}, '
              'ACC: {:.4f}'.format(epoch, args.epochs, epoch_loss, res['mae'],
                                   res['acc']))
        epoch_loss = 0
    return model


def train_step(model, optim, batch, unzip, args):
    """Training on one minibatch.

    Parameters
    ----------
    model
        The model.
    optim
        The optimizer.
    batch : tuple
        Minibatch: :code:`(ob, gt)`. Ob and gt share similar structure as
        below: :code:`{'tau'/'t': {<attr_name>: tensor}}`.
    unzip : function
        A function to unpack the minibatch properly.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    loss : float
        Loss on this minibatch.

    """

    model.train()
    optim.zero_grad()
    ob, gt = unzip(batch, args)
    pred = model(ob, gt)
    # Calculate loss
    loss = loss_calculation(pred, gt, args)
    # Backward propagation
    loss.backward()
    # Update model parameters
    optim.step()
    return loss.item()


def loss_calculation(pred, gt, args):
    """Loss calculation.

    Parameters
    ----------
    pred : :class:`torch.Tensor`
        Model predictions: :code:`{'t1': regression result, 'm1':
        classification result}`.
    gt : :class:`torch.Tensor`
        Prediction ground truth:
        :code:`{'t1': regression ground truth, 'm1': classification result}`.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    loss : :class:`torch.Tensor`
        Training loss.

    """

    regression_loss = regression_loss_calculation(pred, gt, args) \
        if 't1' in pred.keys() else 0
    pred = - torch.gather(pred['m1'], -1, gt['m1'])
    classification_loss = pred.masked_select(gt['m1'] != 0).mean()
    return 5 * classification_loss + regression_loss


def regression_loss_calculation(pred, gt, args):
    """Calculate regression loss.

    Parameters
    ----------
    pred : :class:`torch.Tensor`
        Model predictions: :code:`{'t1': regression result, 'm1':
        classification result}`.
    gt : :class:`torch.Tensor`
        Prediction ground truth:
        :code:`{'t1': regression ground truth, 'm1': classification result}`.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    loss : :class:`torch.Tensor`
        Regression loss of pred['t1'].

    """

    pred, truth, mask = pred['t1'], gt['t1'], (gt['t1'] != 0)
    truth = truth.transpose(0, 1).transpose(1, 2)  # (N, 1, max_len)
    pred = pred.transpose(0, 1).transpose(1, 2)
    tril = torch.tril(torch.ones(pred.size(2), pred.size(2),
                      dtype=torch.bool)).unsqueeze(0).to(args.device)
    pred = torch.sum(pred * tril, 2).transpose(0, 1).unsqueeze(-1)
    truth = torch.sum(truth * tril, 2).transpose(0, 1).unsqueeze(-1)
    pred, truth = pred.masked_select(mask), truth.masked_select(mask)
    loss = F.l1_loss(pred, truth)
    return loss


# =============================================================================
# ============================== Evaluate model ===============================
# =============================================================================
def evaluate(model, dataloader, unzip, args):
    """Evaluation procedure.

    Parameters
    ----------
    model
        The model.
    dataloader : :class:`torch.utils.data.DataLoader`
        Dataloader for dataset to evaluate.
    unzip : function
        A function to unpack the minibatch properly.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    res : dict
        Evaluation results:
        :code:`{'MAE': value, 'RMSE': value, 'ACC': value}`.

    """

    model.eval()
    args.generator = True
    res = {'mae': torch.tensor([0., 0.]), 'rmse': torch.tensor([0., 0.]),
           'acc': torch.tensor([0., 0.])}
    with torch.no_grad():
        for batch in dataloader:
            step_res = evaluate_step(model, batch, unzip, args)
            res = {key: val + step_res[key] for key, val in res.items()}
    args.generator = False
    res = {key: (torch.sqrt(val[0] / val[1])).item()
           if key == 'rmse' else (val[0] / val[1]).item()
           for key, val in res.items()}
    return res


def evaluate_step(model, batch, unzip, args):
    """Evaluation on one minibatch.

    Parameters
    ----------
    model : :class:`torch.nn.Module`
        The model.
    batch : tuple
        Minibatch: :code:`(ob, gt)`. Ob and gt share similar structure as
        below: :code:`{'tau'/'t': {<attr_name>: tensor}}`.
    unzip : function
        A function to unpack the minibatch properly.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    res : dict
        Evaluation results:
        :code:`{'MAE': (loss, # of points), 'RMSE': (loss, # of points),
        'ACC': (# of correct prediction, # of all samples)}`.

    """

    ob, gt = unzip(batch, args)
    pred = model(ob, gt)
    return collect_results(pred, gt, args)


def collect_results(pred, gt, args):
    """Collect results and ground truth for evaluation.

    Parameters
    ----------
    pred : :class:`torch.Tensor`
        Model outputs, a tensor of shape (seq_len, batch, 1).
    gt: :class:`torch.Tensor`
        Ground truth, a tensor of shape (seq_len, batch, 1).
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    mae : :class:`torch.Tensor`
        [loss, # of points]
    mse : :class:`torch.Tensor`
        [loss, # of points]
    acc : :class:`torch.Tensor`
        [# of correct predictions, # of predictions]

    """

    pred_t, truth, mask = pred['t1'], gt['t1'], (gt['t1'] != 0)
    truth = truth.transpose(0, 1).transpose(1, 2)
    pred_t = pred_t.transpose(0, 1).transpose(1, 2)
    tril = torch.tril(torch.ones(pred_t.size(2), pred_t.size(2),
                      dtype=torch.bool)).unsqueeze(0).to(args.device)
    pred_t = torch.sum(pred_t * tril, 2).transpose(0, 1).unsqueeze(-1)
    truth = torch.sum(truth * tril, 2).transpose(0, 1).unsqueeze(-1)
    pred_t, truth = pred_t.masked_select(mask), truth.masked_select(mask)
    mae = torch.tensor([F.l1_loss(pred_t, truth, reduction='sum').item(),
                        pred_t.size(0)])
    mse = torch.tensor([F.mse_loss(pred_t, truth, reduction='sum').item(),
                        pred_t.size(0)])
    pred_m, truth, mask = pred['m1'], gt['m1'], (gt['m1'] != 0)
    pred_m, truth = pred_m.masked_select(mask), truth.masked_select(mask)
    acc = torch.tensor([(pred_m == truth).sum(), pred_m.size(0)])
    return {'mae': mae, 'rmse': mse, 'acc': acc}
