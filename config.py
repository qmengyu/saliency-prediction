work_dir = 'test'

unif_size = (288, 384)

# train setting
lr = 1e-5
epoch_nums = 200

val_step = 5
seed = 1218

reload = False
fine_tune = False


weight_decay = 0
lr_scheduler = dict(type='StepLR', warmup_epochs=10, step_size=3, gamma=0.1)

train_batch_size = 8
val_batch_size = 8

flag = 1