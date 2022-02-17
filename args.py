import argparse

import torch


class Args(object):
    def __init__(self):
        super(Args, self).__init__()
        self.args = argparse.ArgumentParser()

    def parse_args(self):
        """Parse arguments."""

        self.configure_device()
        self.configure_data()
        self.configure_model()
        self.configure_optimizer()
        self.configure_training()
        args = self.args.parse_args()
        if args.cpu:
            args.device = torch.device('cpu')
        elif args.gpu:
            args.device = torch.device('cuda:' + args.use_cuda)
        args = self.set_checkpoint_path(args)
        return args

    def configure_device(self):
        """Device-related configurations."""

        grp = self.args.add_argument_group('Device', 'Config devices.')
        mode = grp.add_mutually_exclusive_group(required=True)
        mode.add_argument('--gpu', action='store_true', help='Use single GPU.')
        mode.add_argument('--cpu', action='store_true', help='Use CPUs.')
        grp.add_argument('--use-cuda', type=str, default='0',
                         help='Which GPU to use?')
        return grp

    def configure_data(self):
        """Data-related configurations."""

        grp = self.args.add_argument_group('Data',
                                           'Data-related configurations.')
        grp.add_argument('--config', type=str, required=True,
                         help='Path to data config file.')
        grp.add_argument('--ob-ratio', type=float, default=0.8,
                         help='Ratio of sequence as observation side.')
        return grp

    def configure_model(self):
        """Model-related configurations."""

        grp = self.args.add_argument_group('Model',
                                           'Model-related configurations.')
        grp.add_argument('--t-emb-size', type=int, default=32,
                         help='Size of the embedded timestamp.')
        grp.add_argument('--e-emb-size', type=int, default=32,
                         help='Size of the embedded event type.')
        grp.add_argument('--hidden-size', type=int, default=16,
                         help='Size of the hidden state of RNNs.')
        grp.add_argument('--dropout', type=float, default=0.0,
                         help='Dropout rate.')
        grp.add_argument('--n-heads', type=int, default=4,
                         help='d_h')
        grp.add_argument('--m1', type=int, default=4, help='m_1')
        grp.add_argument('--m2', type=int, default=2, help='m_2')
        grp.add_argument('--dq', type=int, default=4,
                         help='Dimension of head-wise query.')
        grp.add_argument('--dk', type=int, default=4,
                         help='Dimension of head-wise key.')
        grp.add_argument('--dv', type=int, default=4,
                         help='Dimension of head-wise value.')
        return grp

    def configure_optimizer(self):
        """Optimizer-related configurations."""

        grp = self.args.add_argument_group('optimizer',
                                           'Optimizer-related configurations.')
        grp.add_argument('--lr', type=float, default=0.0001,
                         help='Learning rate.')
        grp.add_argument('--weight-decay', type=float, default=0.0,
                         help='Weight decay for optimizer.')
        return grp

    def configure_training(self):
        """Training process related parameters."""

        grp = self.args.add_argument_group('Training',
                                           'Training process related '
                                           'parameters.')
        grp.add_argument('--epochs', type=int, default=10,
                         help='# of epochs for training.')
        grp.add_argument('--batch-size', type=int, default=128,
                         help='Size of minibatch.')
        return grp

    def set_checkpoint_path(self, args):
        """Set up checkpoint path."""

        device = str(args.device)
        args.checkpoint = ('.').join(['checkpoint', device, 'pth'])
        signature = '[dma-nets]'
        args.checkpoint = args.checkpoint[:-3] + signature + '.pth'
        return args


args = Args().parse_args()
print(args)
