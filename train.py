from metrics.salience_metrics import nss_loss, corr_coeff_loss, kld_loss, loss_similarity
from utils.utils import build_scheduler, save_checkpoint


class Trainer:
    def __init__(self, lr, dataloder, work_dir, device, start_epoch, epoch_nums, val_step,
                writer):
        self.lr = lr
        self.dataloder = dataloder
        self.work_dir = work_dir
        self.device = device
        self.cur_epoch = start_epoch
        self.epoch_nums = epoch_nums
        self.val_step = val_step
        self.writer = writer

    def train_epochs(self, model, optimizer, lr_scheduler, evaluation,):
        model.train()
        if lr_scheduler:
            scheduler = build_scheduler(lr_scheduler=lr_scheduler, optimizer=optimizer)

        for self.cur_epoch in range(self.cur_epoch, self.epoch_nums + 1):
            model.train()
            train_performance, batch_count = {'loss': 0}, 0
            for i_batch, batch in enumerate(self.dataloder):
                imgs, sals, fix_maps = \
                    batch["imgs"].to(self.device), batch["sals"].to(self.device), batch["fix_maps"].to(self.device)

                optimizer.zero_grad()

                preds = model(imgs)
                loss = 10 * kld_loss(preds, sals) - 1 * nss_loss(preds, fix_maps).mean() - 2 * corr_coeff_loss(preds, sals).mean()
                - 1 * loss_similarity(preds, sals)

                batch_count += 1
                loss_item = loss.detach().cpu().item()
                train_performance['loss'] += loss_item
                print(f'train_(epoch{self.cur_epoch:3d}/{i_batch:4d}), ' + f'loss: {loss_item:.3f}')
                loss.backward()
                optimizer.step()

            train_performance['loss'] /= batch_count
            self.writer.add_scalar('AA_Scalar/train_loss', train_performance['loss'], self.cur_epoch)
            self.writer.add_scalar('AA_Scalar/train_lr', float(optimizer.param_groups[0]['lr']), self.cur_epoch)

            if lr_scheduler:
                scheduler.step()

            if self.cur_epoch % self.val_step == 0:
                evaluation.validation(model, self.cur_epoch, writer=self.writer,
                                      dataset_name='salicon', phase="val", savePath=False)

                save_checkpoint(self.cur_epoch, model, optimizer, self.work_dir)
