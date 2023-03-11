import collections

from PIL import Image
from numpy import mean
from torchvision.transforms import transforms

from datasets.dataset import SaliencyDataloder
from metrics.salience_metrics import *
from utils.utils import *

salgan_path = "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/SALICON/saliency_maps/SalGAN"
val_images_path = ""

salicon_val_dataloder = SaliencyDataloder(source='salicon', batch_size=1, phase='val',
                                                   out_size=(192, 256), seed=0)

dataloder = salicon_val_dataloder.get_dataloder()
dataset = salicon_val_dataloder.get_dataset()
other_maps = dataset.get_other_maps()


results = collections.defaultdict(list)
for i_batch, batch in enumerate(dataloder):
    imgs, sals, fix_maps = \
        batch["imgs"], batch["sals"], batch["fix_maps"]

    sal, fix_map, ori_size, file_name = \
        sals.squeeze(), fix_maps.squeeze(), \
        batch["ori_sizes"].squeeze(), batch["file_names"][0]

    # preds = log_softmax(preds).squeeze()

    pred = Image.open(os.path.join(salgan_path, file_name+'.jpg'))
    pred = transforms.ToTensor()(pred)
    pred = normalize_tensor(pred, zero_fill=True).log().squeeze()

    cur_result = {"kld": kld_loss(pred, sal).cpu().item(),
                  "nss": nss_loss(pred.exp(), fix_map).cpu().item(),
                  "cc": corr_coeff_loss(pred.exp(), sal).cpu().item()}

    pred = pred.exp().cpu().numpy()

    cur_result['sim'] = similarity(pred, sal.cpu().numpy())
    cur_result['aucj'] = auc_judd(pred, fix_map.int().cpu().numpy())
    other_map = next(other_maps)
    cur_result['aucs'] = auc_shuff_acl(pred, fix_map.cpu().numpy(), other_map)

    for metirc, score in cur_result.items():
        results[metirc].append(score)

    print(f'val_/{i_batch:4d}), ' +
          ','.join(f' {metirc:4s}: {score:.3f}' for metirc, score in cur_result.items()))

for metric, all_res in results.items():
    print(f'val_{metric}', mean(all_res))

