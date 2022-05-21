import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='./data',
                        help='root path to data directory')
    parser.add_argument('-s', '--source-names', type=str, required=True, default='veri', nargs='+',
                        help='source dataset for training(delimited by space)')
    parser.add_argument('-t', '--target-names', type=str, required=True, default='veri', nargs='+',
                        help='target dataset for testing(delimited by space)')
    parser.add_argument('-j', '--workers', default=1, type=int,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    # split-id not used
    parser.add_argument('--split-id', type=int, default=0,
                        help='split index (note: 0-based)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=256,
                        help='width of an image')
    parser.add_argument('--train-sampler', type=str, default='RandomIdentitySampler', choices=('RandomSampler', 'RandomIdentitySampler'),
                        help='sampler for trainloader')

    # ************************************************************
    # Data augmentation
    # ************************************************************
    parser.add_argument('--random-erase', default=True, type=bool,
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', default=False, type=bool,
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', default=False, type=bool,
                        help='randomly alter the intensities of RGB channels')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimization algorithm (see optimizers.py)')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help='weight decay')
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help='exponential decay rate for adam\'s second moment')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=80, type=int,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')

    parser.add_argument('--train-batch-size', default=28, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=28, type=int,
                        help='test batch size')

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step', choices=['single_step', 'multi_step', 'warm_up',
                                                                                   'cosine_step', 'warmup_cosine',
                                                                                   'warmup_cosine_step', 'cyclic_cosine',
                                                                                    'warmup_cosine_cosine'],
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[20, 40, 60], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--step_epoch', default=20, type=int,
                        help='step epoch to decay learning rate using cosine annealing')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--label-smooth', type=bool, default=True,
                        help='use label smoothing regularizer in cross entropy loss')
    # ************************************************************
    # Circle classifier setting
    # ************************************************************
    parser.add_argument('--circle_classifier', type=bool, default=False, help='whether'
                        'using circle classifier instead of linear one')
    parser.add_argument('--cosine_scale', type=int, default=64, help='cosine sacle in circle classifier')
    parser.add_argument('--cosine_margin', type=float, default=0.35, help='cosine margin in circle classifier')
    parser.add_argument('--gem_pooling', type=bool, default=False, help='whether'
                        'using gem pooling layer')
    parser.add_argument('--syn_bn', type=bool, default=False, help='whether'
                        'using synchronized batch normalization')
    parser.add_argument('--use_apex', type=bool, default=True, help='whether'
                        'using apex to accelerate training')

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.5,
                        help='margin for triplet loss')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--lambda-xent', type=float, default=1.0,
                        help='weight to balance cross entropy loss')
    parser.add_argument('--lambda-htri', type=float, default=1.0,
                        help='weight to balance hard triplet loss')
    parser.add_argument('--lambda-xent_rs', type=float, default=0.0,
                        help='weight to balance cross entropy loss of rotation space')
    parser.add_argument('--lambda-htri_rs', type=float, default=0.0,
                        help='weight to balance hard triplet loss of rotation space')
    parser.add_argument('--lambda-kp-global', type=list, default=[1.0, 1.0],
                        help='weight to balance key-points branch and global branch')
    parser.add_argument('--lambda-rot', type=float, default=1.0,
                        help='weight to balance rotation prediction task loss')
    parser.add_argument('--lambda-eqv', type=float, default=0.0,
                        help='weight for equivariance constraint between image and rotated image')
    parser.add_argument('--lambda-of', type=float, default=0.0,
                        help='weight for orthogonality of features')

    # ***********************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('--a-description', type=str, default='',
                        help='The description about this project.')
    parser.add_argument('-a', '--arch', type=str, default='resnet50_kp_atten',
                        choices=['resnet50_kp_atten'])

    parser.add_argument('--test-size', default=1600, type=int, choices=(800, 1600, 2400))
    parser.add_argument('--flipped-test', default=False, type=bool,
                        help='evaluate only')
    parser.add_argument('--feat_norm', default=True, type=bool,
                        help='test feature normalization')

    # ***********************************************************
    # Freezing parameters except from ABDModule
    # ************************************************************
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    parser.add_argument('--freeze-start-epoch', type=int, default=10)
    parser.add_argument('--freeze-end-epoch', type=int, default=15)

    # ***********************************************************
    # Rotation prediction setting
    # ************************************************************
    parser.add_argument('--use-rotation-prediction', type=bool, default=True, help='whether'
                        'using image rotation degree prediction task to boost the model')
    parser.add_argument('--rot-start-epoch', type=int, default=-1)
    parser.add_argument('--use-rot-space', type=bool, default=False, help='whether'
                        'using image rotation space')
    # Other setting
    parser.add_argument('--use-equivariance-constraint', default=False, type=bool)
    parser.add_argument('--use-of-penalty', default=False, type=bool)
    parser.add_argument('--start-of-penalty', default=-1, type=int)
    parser.add_argument('--use-residual-in-abdmodul', type=bool, default=True)
    # ***********************************************************
    # Testing set
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that don\'t match in size')
    parser.add_argument('--evaluate', default=True, type=bool,
                        help='evaluate only')
    parser.add_argument('--visualize-ranks', default=True, type=bool,
                        help='visualize ranked results, only available in evaluation mode')
    parser.add_argument('--vis-landmark-mask', default=True, type=bool,
                        help='visualize landmark mask')


    parser.add_argument('--of-beta', type=float, default=1e-6)
    parser.add_argument('--rotation', type=int, default=4, choices=(4,), help='number of classes of rotation prediction')
    parser.add_argument('--lable-smooth-for-rot', type=bool, default=False)
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')
    parser.add_argument('--freezing_kp_branch', default=False,
                        help='freeze the weights of keypoints extraction branch')
    parser.add_argument('--freezing_kp_branch_always', default=False,
                        help='freeze the weights of keypoints extraction branch always')
    parser.add_argument('--kp_branch_start_epoch', default=parser.parse_known_args()[0].stepsize[-1],
                        help='fine tune the keypoints branch from this epoch')

    # ************************************************************
    # ResNet50_Kp_Concate setting
    # ************************************************************
    parser.add_argument('--use-concat-or-multip', type=str, default='concat', choices=('concat', 'multip'),
                        help='the manner of using detected landmarks')

    # ************************************************************
    # ABD architecture
    # ************************************************************
    parser.add_argument('--abd-dim', type=int, default=1024)
    parser.add_argument('--abd-dan', type=tuple, default=('cam', 'pam'))
    parser.add_argument('--abd-dan-no-head', action='store_true')

    # ************************************************************
    # Test settings
    # ************************************************************

    parser.add_argument('--eval-freq', type=int, default=2,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--query-remove', type=bool, default=True)
    parser.add_argument('--max-rank', type=int, default=50)
    # ************************************************************
    # Image-to-track test settings
    # ************************************************************
    parser.add_argument('--image2track-test', type=bool, default=True)
    parser.add_argument('--gallery-name-list', type=str, default='./data/VeRi/name_test.txt')
    parser.add_argument('--gallery-track-list', type=str, default='./data/VeRi/test_track.txt')
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=3,
                        help='manual seed: 1,2,3')
    parser.add_argument('--save-dir', type=str, default='./log',
                        help='path to save log and model weights')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='3,4', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices (useful when using managed clusters)')
    return parser


def dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
        'rotation': parsed_args.rotation
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in lr_schedulers.py from
    the parsed command-line arguments.
    """
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
        'max_epoch': parsed_args.max_epoch,
        'step_epoch': parsed_args.step_epoch,
    }
