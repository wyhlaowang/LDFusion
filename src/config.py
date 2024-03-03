import time
import argparse
from sys import platform

parser = argparse.ArgumentParser(description='UniFusion')


# Dataset yes
parser.add_argument('--file_path',
                    type=str,
                    # default='C:/Users/wyh/Desktop/FusionData/U2Fusion/Training_dataset/vis_ir_dataset64.h5',
                    # default='C:/Users/wyh/Desktop/FusionData/CUFD/vis_ir_orig.h5',
                    default='/home/hosthome/Datasets/vis_ir_orig.h5',
                    # default='/home/hosthome/Datasets/vis_ir_dataset64.h5',
                    help='path to h5 file')
parser.add_argument('--height',
                    type=int,
                    default=84,
                    help='height of a raw image')
parser.add_argument('--width',
                    type=int,
                    default=84,
                    help='width of a raw image')
parser.add_argument('--crop_height',
                    type=int,
                    default=84,
                    help='height of a patch to crop')
parser.add_argument('--crop_width',
                    type=int,
                    default=84,
                    help='width of a patch to crop')
parser.add_argument('--if_augment',
                    type=bool,
                    default=False,
                    help='augment or not')
parser.add_argument('--if_test',
                    type=bool,
                    default=False,
                    help='test or not(only for visir)')
parser.add_argument('--test_size',
                    type=int,
                    default=1000,
                    help='size of test dataset(only for visir)')

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
                    default=8 if platform == 'linux' else 2,
                    help='number of threads')
parser.add_argument('--local_rank',
                    type=int,
                    help='local gpu id')

# Network yes
parser.add_argument('--model_name',
                    type=str,
                    default='UniFusion',
                    choices=('UniFusion'),
                    help='model name')
parser.add_argument('--network',
                    type=str,
                    default='resnet34',
                    choices=('resnet18', 'resnet34'),
                    help='network name')
parser.add_argument('--pretrained',
                    type=bool,
                    default=True,
                    help='load pretrained model')

expansion = int(5)

# Training ing
parser.add_argument('--epochs',
                    type=int,
                    default=expansion*20,
                    help='epochs for training')
parser.add_argument('--train_batch',
                    type=int,
                    default=4 if platform=='linux' else 2,
                    help='batch size for training')
parser.add_argument('--lr',
                    type=float,
                    default=0.0003 if platform=='linux' else 0.0002,
                    help='learning rate for training')
parser.add_argument('--lr_decay_gamma',
                    type=float,
                    default=0.5,
                    help='learning rate dacay gamma value')
parser.add_argument('--lr_mstone',
                    type=list,
                    default=[expansion*i for i in [10,12,14,16,18,20]],
                    help='learning rate decay epoch')
parser.add_argument('--if_ema',
                    type=bool,
                    default=False,
                    help='use ema or not')                    
parser.add_argument('--ema_decay',
                    type=float,
                    default=0.995,
                    help='ema decay')
parser.add_argument('--ema_update_every',
                    type=int,
                    default=10,
                    help='ema update every')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--loader_shuffle',
                    type=bool,
                    default=True,
                    help='shuffle reader or not')
parser.add_argument('--if_split_batch',
                    type=bool,
                    default=False,
                    help='split batch or not')
parser.add_argument('--if_warm_up',
                    type=bool,
                    default=True,
                    help='learning rate warm up during the 1st epoch')
parser.add_argument('--mixed_precision',
                    type=str,
                    default='no',
                    choices=('no', 'fp16', 'bf16'),
                    help='mixed precision mode, bf16 requires pytorch 1.10 or higher')      
parser.add_argument('--if_amp',
                    type=bool,
                    default=False,
                    help='use amp or not')
# Logs ing
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--load',
                    type=str,
                    default='220907_152914_trial',
                    help='file name to load')

args = parser.parse_args()

args.num_gpus = len(args.gpus.split(','))

current_time = time.strftime('%y%m%d_%H%M%S_')
save_dir = './experiments/' + current_time + args.save
args.save_dir = save_dir
load_dir = './experiments/' + args.load
args.load_dir = load_dir
