import collections
import torch
from numpy import mean
from datasets.dataset import SaliencyDataloder
from metrics.salience_metrics import similarity, auc_shuff_acl, corr_coeff_loss, nss_loss, kld_loss, auc_judd, \
    loss_similarity


class Evaluation:
    def __init__(self,batch_size, device, out_size, seed=0):
        self.device = device
        self.mit1003_val_dataloder = SaliencyDataloder(source='mit1003', batch_size=batch_size, phase='val',
                                              out_size=out_size, seed=seed)
        self.mit1003_test_dataloder = SaliencyDataloder(source='mit1003', batch_size=batch_size, phase='test',
                                              out_size=out_size, seed=seed)
        self.mit300_test_dataloder = SaliencyDataloder(source='mit300', batch_size=batch_size, phase='test',
                                                   out_size=out_size, seed=seed)
        self.salicon_val_dataloder = SaliencyDataloder(source='salicon', batch_size=batch_size, phase='val',
                                                   out_size=out_size, seed=seed)

        self.dataloders = {
            'mit1003_val': self.mit1003_val_dataloder,
            'mit1003_test': self.mit1003_test_dataloder,
            'mit300_test': self.mit300_test_dataloder,
            'salicon_val': self.salicon_val_dataloder,
        }


    def validation(self, model, epoch, writer, dataset_name, phase="val", savePath=None):
        model.eval()
        dataloder = self.dataloders[f'{dataset_name}_{phase}'.lower()].get_dataloder()
        dataset = self.dataloders[f'{dataset_name}_{phase}'.lower()].get_dataset()
        other_maps = dataset.get_other_maps()

        with torch.no_grad():
            results = collections.defaultdict(list)
            for i_batch, batch in enumerate(dataloder):
                imgs, sals, fix_maps = \
                    batch["imgs"].to(self.device), batch["sals"].to(self.device), batch["fix_maps"].to(self.device)
                preds = model(imgs)

                loss = 10 * kld_loss(preds, sals) - 1 * nss_loss(preds, fix_maps).mean() - 2 * corr_coeff_loss(preds, sals).mean()
                - 1 * loss_similarity(preds, sals)

                batch_size = 1   # speed the val
                cur_result = collections.defaultdict(int)
                cur_result.update({"loss": loss.detach().cpu().item() * batch_size,
                    "kld": kld_loss(preds, sals).cpu().item() * batch_size,
                              "nss": nss_loss(preds, fix_maps).mean().cpu().item() * batch_size,
                              "cc": corr_coeff_loss(preds, sals).mean().cpu().item() * batch_size,
                            'sim': loss_similarity(preds, sals) .mean().cpu().item() * batch_size})

                for index in range(batch_size):
                    pred, sal, fix_map, ori_size, file_name = \
                        preds[index], sals[index], fix_maps[index], \
                        batch["ori_sizes"][index], batch["file_names"][index]
                    pred = pred.cpu().numpy()
                    cur_result['aucj'] += auc_judd(pred, fix_map.int().cpu().numpy())
                    other_map = next(other_maps)
                    cur_result['aucs'] +=  auc_shuff_acl(pred, fix_map.cpu().numpy(), other_map)

                for metirc, score in cur_result.items():
                    score /= batch_size
                    results[metirc].append(score)

                print(f'val_(epoch{epoch:3d}/{i_batch:4d}), ' +
                      ','.join(f' {metirc:4s}: {score:.3f}' for metirc, score in cur_result.items()))

                if i_batch > 100: break

            for metric, all_res in results.items():
                if metric=="loss":
                    writer.add_scalar('AA_Scalar/val_loss', mean(all_res), epoch)
                else:
                    writer.add_scalar(f'val_{dataset_name}/{metric}', mean(all_res), epoch)
