import datetime
from inspect import getfullargspec
import json
import copy
import sys
import importlib
import time
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
import torch

from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
    ExponentialLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler


def preprocess_size_img(img_np, target_h, target_w, channels=3):
    # img_rgp_np :(H, W, C)

    if channels == 1:
        img_padded = np.zeros((target_h, target_w), dtype=np.uint8)
    else:
        img_padded = np.ones((target_h, target_w, channels), dtype=np.uint8)

    original_shape = img_np.shape
    h_rate = original_shape[0] / target_h
    w_rate = original_shape[1] / target_w

    if h_rate > w_rate:  # 原图比例宽较小，宽度方向填充
        new_w = (original_shape[1] * target_h) // original_shape[0]
        img = cv2.resize(img_np, (new_w, target_h))  # 调整原图比例
        if new_w > target_w:
            new_w = target_w
        img_padded[:,
        ((img_padded.shape[1] - new_w) // 2):((img_padded.shape[1] - new_w) // 2 + new_w)] = img
    else:
        new_h = (original_shape[0] * target_w) // original_shape[1]
        img = cv2.resize(img_np, (target_w, new_h))

        if new_h > target_h:
            new_h = target_h
        img_padded[((img_padded.shape[0] - new_h) // 2):((img_padded.shape[0] - new_h) // 2 + new_h),
        :] = img

    return img_padded

# test
def preprocess_size_fixmap(fix_map, target_h, target_w):
    # fix_map :(H, W)
    fix_padded = np.zeros((target_h, target_w), dtype=np.uint8)

    original_shape = fix_map.shape
    h_rate = original_shape[0] / target_h
    w_rate = original_shape[1] / target_w
    indexs_h, indexs_w = np.where(fix_map > 200)

    if h_rate > w_rate:  # 原图比例宽较小，宽度方向填充, 适配目标高度
        indexs_h, indexs_w = (indexs_h / h_rate).astype(np.uint8), (indexs_w / h_rate).astype(np.uint8)
        new_w = (original_shape[1] * target_h) // original_shape[0]
        indexs_w += (target_w - new_w) // 2
    else:
        indexs_h, indexs_w = (indexs_h / w_rate).astype(np.uint8), (indexs_w / w_rate).astype(np.uint8)
        new_h = (original_shape[0] * target_w) // original_shape[1]
        indexs_h += (target_h - new_h) // 2
    fix_padded[(indexs_h, indexs_w)] = 255
    return fix_padded


def postprocess_size_img(pred, org_h, org_w):

    pred = np.array(pred)
    predictions_shape = pred.shape
    h_rate = org_h / predictions_shape[0]
    w_rate = org_w / predictions_shape[1]

    if h_rate > w_rate:  # 原图比例宽较小，由宽度方向填充而来
        new_w = (predictions_shape[1] * org_h) // predictions_shape[0]
        pred = cv2.resize(pred, (new_w, org_h))
        img = pred[:, ((pred.shape[1] - org_w) // 2):((pred.shape[1] - org_w) // 2 + org_w)]
    else:
        new_h = (predictions_shape[0] * org_w) // predictions_shape[1]
        pred = cv2.resize(pred, (org_w, new_h))
        img = pred[((pred.shape[0] - org_h) // 2):((pred.shape[0] - org_h) // 2 + org_h), :]

    return img


def get_kwargs_names(func):
    args = getfullargspec(func).args
    try:
        args.remove('self')
    except ValueError:
        pass
    return args


def get_kwargs_dict(obj):
    return {key: obj.__dict__[key]
            for key in get_kwargs_names(obj.__init__)}


class KwConfigClass:

    def asdict(self):
        return get_kwargs_dict(self)

    @classmethod
    def init_from_cfg_dir(cls, directory, **kwargs):
        with open(directory / f"{cls.__name__}.json", 'r') as f:
            config = json.load(f)
        if 'new_instance' in config:
            config['new_instance'] = False
        config.update(kwargs)
        return cls(**config)

    def save_cfg(self, directory):
        with open(directory / f"{self.__class__.__name__}.json", 'w') as f:
            json.dump(self.asdict(), f)


def get_timestamp():
    return str(datetime.datetime.now())[:-7].replace(' ', '_')


def load_module(directory, module_name, full_name):
    path = copy.deepcopy(sys.path)
    sys.path.insert(0, str(directory))
    if module_name is not None and module_name in sys.modules:
        del sys.modules[module_name]
    if full_name in sys.modules:
        del sys.modules[full_name]
    module = importlib.import_module(full_name)
    sys.path = path
    return module


def load_trainer(directory, **kwargs):
    train = load_module(directory / 'code_copy', 'bds_net', 'bds_net.train')
    print(train)
    train = train.Trainer.init_from_cfg_dir(directory, **kwargs)
    return train


def load_model(directory, **kwargs):
    # model = load_module(directory, 'code_copy', 'code_copy.model')
    model = load_module(directory / 'code_copy', 'bds_net', 'bds_net.model')
    print(model)
    model_cls = model.get_model()
    model = model_cls.init_from_cfg_dir(directory, **kwargs)
    return model


def load_dataset(directory, **kwargs):
    # data = load_module(directory, 'code_copy', 'code_copy.data')
    data = load_module(directory / 'code_copy', 'bds_net', 'bds_net.data')
    print(data)
    dataset_cls = data.get_dataset()
    dataset = dataset_cls.init_from_cfg_dir(directory, **kwargs)
    return dataset


class Timer:
    def __init__(self, name='', info='', verbose=True):
        self.name = name
        self.verbose = verbose
        self.since = time.time()
        if name and self.verbose:
            print(name + ' ' + info + '...')

    def finish(self):
        time_elapsed = time.time() - self.since
        if self.verbose:
            print('{} completed in {:.0f}m {:.0f}s'.format(
                self.name, time_elapsed // 60, time_elapsed % 60))
        return time_elapsed


def normalize_tensor(tensor, rescale=False, zero_fill=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin

    if zero_fill:
        tensor = torch.where(tensor == 0, tensor.max() * 1e-4, tensor)
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor


def normalize_array(array, rescale=False):
    amin = np.amin(array)
    if rescale or amin < 0:
        array -= amin
    asum = array.sum()
    if asum > 0:
        return array / asum
    print("Zero array")
    array.fill(1. / array.size())
    return array


def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)


def nss(pred, fixations):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    results = results.reshape(size[:2])
    return results


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16,
                            np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                            np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def smooth_sequence(seq, method):
    shape = seq.shape

    seq = seq.reshape(shape[1], np.prod(shape[-2:]))
    if method[:3] == 'med':
        kernel_size = int(method[3:])
        ks2 = kernel_size // 2
        smoothed = np.zeros_like(seq)
        for idx in range(seq.shape[0]):
            smoothed[idx, :] = np.median(
                seq[max(0, idx - ks2):min(seq.shape[0], idx + ks2 + 1), :],
                axis=0)
        seq = smoothed.reshape(shape)
    else:
        raise NotImplementedError

    return seq


def show_tensor_heatmap(img, annot=None, fmt=".1f", save_path=None):
    plt.figure(figsize=(10, 20))  # 画布大小
    sns.set()
    ax = sns.heatmap(img, cmap="rainbow", annot=annot, fmt=fmt)  # cmap是热力图颜色的参数

    # plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_str_file(save_path, str0):
    filename = open(save_path, 'w')
    filename.write(str0)
    filename.close()


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)


def save_checkpoint(epoch_num, model, optimizer, work_dir):
    checkpointName = 'ep{}.pth.tar'.format(epoch_num)
    checkpointpath = f'{work_dir}/checkpoint/'
    if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath)
    checkpoint = {
        'epoch': epoch_num,
        'model': model.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, os.path.join(checkpointpath, checkpointName))


def loadCheckpoint(epoch, model, optimizer, work_dir, load_best=False):
    if load_best:
        model_dir_name = f'model_weights/best_model_and_config/'
        checkpointName = f'best_model.pth.tar'
        checkpointPath = os.path.join(model_dir_name, checkpointName)
    else:
        model_dir_name = f'{work_dir}/checkpoint/'
        if not os.path.exists(model_dir_name):
            os.mkdir(model_dir_name)

        model_dir = os.listdir(model_dir_name)  # 列出文件夹下文件名
        if len(model_dir) == 0:
            return 0, model, optimizer
        model_dir.sort(key=lambda x: int(x[2:-8]))  # 文件名按数字排序
        if epoch == -1:
            checkpointName = model_dir[epoch]  # 获取文件 , epoch = -1 获取最后一个文件
        else:
            checkpointName = f'ep{epoch}.pth.tar'
        checkpointPath = os.path.join(model_dir_name, checkpointName)

    if os.path.isfile(checkpointPath):
        print(f"Loading {checkpointPath}...")
        checkpoint = torch.load(checkpointPath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.param_groups[0]['lr'] = checkpoint['lr']
        print('Checkpoint loaded')
    else:
        raise OSError('Checkpoint not found')

    return checkpoint['epoch'], model, optimizer


def build_scheduler(optimizer, lr_scheduler):
    name_scheduler = lr_scheduler.type
    scheduler = None

    if name_scheduler == 'StepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = StepLR(optimizer=optimizer, step_size=lr_scheduler.step_size, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=lr_scheduler.T_max)
    elif name_scheduler == 'ReduceLROnPlateau':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step(val_loss)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=lr_scheduler.mode)
    elif name_scheduler == 'LambdaLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler.lr_lambda)
    elif name_scheduler == 'MultiStepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=lr_scheduler.milestones, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CyclicLR':
        # >>> for epoch in range(10):
        # >>>   for batch in data_loader:
        # >>>       train_batch(...)
        # >>>       scheduler.step()
        scheduler = CyclicLR(optimizer=optimizer, base_lr=lr_scheduler.base_lr, max_lr=lr_scheduler.max_lr)
    elif name_scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingWarmRestarts':
        # >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        # >>> for epoch in range(20):
        #     >>> scheduler.step()
        # >>> scheduler.step(26)
        # >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=lr_scheduler.T_0,
                                                T_mult=lr_scheduler.T_mult)

    if lr_scheduler.warmup_epochs != 0:
        scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=lr_scheduler.warmup_epochs, after_scheduler=scheduler)

    if scheduler is None:
        raise Exception('scheduler is wrong')
    return scheduler
