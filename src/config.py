import time
import argparse

parser = argparse.ArgumentParser(description='IVIF_Fusion')


# Dataset yes
parser.add_argument('--crop_height',
                    type=int,
                    default=224,
                    help='height of a patch to crop')
parser.add_argument('--crop_width',
                    type=int,
                    default=224,
                    help='width of a patch to crop')
parser.add_argument('--if_transform',
                    type=bool,
                    default=True,
                    help='augment or not')
parser.add_argument('--if_pair',
                    type=bool,
                    default=True,
                    help='test or not(only for visir)')


# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=3407,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default='0',
                    help='visible GPUs')
parser.add_argument('--num_workers',
                    type=int,
                    default=32,
                    help='number of threads')
parser.add_argument('--local_rank',
                    type=int,
                    help='local gpu id')

expansion = int(5)
# Training ing
parser.add_argument('--epochs',
                    type=int,
                    default=expansion*20,
                    help='epochs for training')
parser.add_argument('--train_batch',
                    type=int,
                    default=16,
                    help='batch size for training')
parser.add_argument('--lr',
                    type=float,
                    default=0.0005,
                    help='learning rate for training')
parser.add_argument('--lr_decay_gamma',
                    type=float,
                    default=0.5,
                    help='learning rate dacay gamma value')
parser.add_argument('--lr_mstone',
                    type=list,
                    default=[expansion*i for i in [10,12,14,16,18,20]],
                    help='learning rate decay epoch')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--if_warm_up',
                    type=bool,
                    default=True,
                    help='learning rate warm up during the 1st epoch')   

# Logs ing
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')


args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = './experiments/' + current_time + args.save
args.save_dir = save_dir

