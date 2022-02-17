import torch

from args import args
from init import init
from data import unpacking_s2s
from learn import train, evaluate


def training(model, optim, train_set, valid_set, unzip, args):
    """Training procedure.

    Parameters
    ----------
    model : :class:`torch.nn.Model`
        The initialized model.
    optim : :class:`torch.Optimizer`
        The optimizer.
    train_set : :class:`torch.DataLoader`
        Training dataset.
    valid_set : :class:`torch.DataLoader`
        Validation dataset.
    unzip :
        Function that unpacks the minibatch.
    args : :class:`argparse.Namespace`
        Configurations.

    Returns
    -------
    model : :class:`torch.nn.Model`
        The trained model.

    """

    model = train(model, optim, train_set, valid_set, unzip, args)
    torch.save({'model': model.state_dict()}, args.checkpoint)
    return model


def evaluation(model, train_set, test_set, unzip, args):
    """Evaluation procedure.

    Parameters
    ----------
    model : :class:`torch.nn.Model`
        The initialized model.
    train_set : :class:`torch.DataLoader`
        Training dataset.
    test_set : :class:`torch.DataLoader`
        Test dataset.
    unzip :
        Function that unpacks the minibatch.
    args : :class:`argparse.Namespace`
        Configurations.

    """

    saved_model = torch.load(args.checkpoint)
    return evaluation_on_dataset(model, saved_model, test_set, unzip, args)


def evaluation_on_dataset(model, saved_model, dataloader, unzip, args):
    """Evaluation on a dataset.

    Parameters
    ----------
    model : :class:`torch.nn.Model`
        The initialized model.
    models : dict
        Saved models.
    dataloader : :class:`troch.DataLoader`
        Dataset.
    unzip : function
        Function that unpacks the minibatch.
    args : :class:`argparse.Namespace`
        Configurations.

    """

    res = {'mae': 0, 'rmse': 0, 'acc': 0}
    model.load_state_dict(saved_model['model'])
    res = {key: val
           for key, val in evaluate(model, dataloader, unzip, args).items()}
    print('MAE: {:.4f}, RMSE: {:.4f}, ACC: {:.4f}'.format(
        res['mae'], res['rmse'], res['acc']))


tr_set, val_set, te_set, model, optim = init(args)
model = training(model, optim, tr_set, val_set, unpacking_s2s, args)
evaluation(model, tr_set, te_set, unpacking_s2s, args)
