import argparse
import os
from mmcv import DictAction, Config
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import SaliencyDataloder
from eval import Evaluation

from train import Trainer
from utils.utils import setup_seed, loadCheckpoint

parser = argparse.ArgumentParser(description='Train a models')
parser.add_argument('--config', default='./config.py', help='config.py path')
parser.add_argument('--work_dir', default='test', help='path to save logs and weights')
parser.add_argument('--device', default='cuda:3', help='cuda:n')
parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

args = parser.parse_args()
cfg = Config.fromfile(args.config)
cfg.merge_from_dict(vars(args))

if args.options is not None:
    cfg.merge_from_dict(args.options)

cfg.work_dir = os.path.join('/data/qmengyu/02-Results/02-Saliency-Dataset/logs/', cfg.work_dir)

writer = SummaryWriter(log_dir=cfg.work_dir)
setup_seed(cfg.seed)

if cfg.flag:
    from models.TranSalNet_Res import TranSalNet
else:
    from models.TranSalNet_Dense import TranSalNet
model = TranSalNet().to(cfg.device)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

if cfg.reload:
    epoch_start, model, optimizer = loadCheckpoint(-1, model, optimizer, cfg.work_dir)
elif cfg.fine_tune:
    epoch_start, model, optimizer = loadCheckpoint(-1, model, optimizer, None, load_best=True)
else:
    epoch_start = 0


print('model training results will be in :',cfg.work_dir)

cfg.lr *= cfg.train_batch_size  # test

cfg.dump(os.path.join(cfg.work_dir, 'config.py'))
# save_str_file(f'{cfg.work_dir}/config.py', cfg.pretty_text)


train_dataloder = SaliencyDataloder(source='salicon', batch_size=cfg.train_batch_size, phase='train',
                                          out_size=cfg.unif_size, seed=cfg.seed).get_dataloder()

train = Trainer(lr=cfg.lr, dataloder=train_dataloder, work_dir=cfg.work_dir, device=cfg.device,
                start_epoch=epoch_start, epoch_nums=cfg.epoch_nums, val_step=cfg.val_step,
                writer=writer)

evaluation = Evaluation(batch_size=cfg.val_batch_size, device=cfg.device, out_size=cfg.unif_size, seed=cfg.seed)

train.train_epochs(model, optimizer, lr_scheduler=cfg.lr_scheduler, evaluation=evaluation)


