# lesion level foundation model pretraining
# run on deeplesion with single GPU.
import torch
from lightly.loss import MSNLoss
from lightly.models import utils
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightly.data.dataset import LightlyDataset
from msn_custom_CT_transform import MSNTransform_CT
from lightly.utils.debug import std_of_l2_normalized
import os
import time
from utils import WarmupCosineSchedule, CosineWDSchedule
from msn_custom_ViT_model import MSN, vit
from PIL import Image
import numpy as np

# training results save
results_save_folder = "./ckps/LFM_pretraining/"
os.makedirs(results_save_folder, exist_ok=True)

path_to_data = "/data/DeepLesion/lesion_patches/"

# lesion patch size
image_size = 64

# ensure reproducabilty
seed = 666
pl.seed_everything(seed)

# hyper-parameters
batch_size = 256
num_epochs = 500

warmup_epochs = 15
start_lr = 2e-4
ref_lr = 0.001
final_lr = 1e-6
weight_decay = 0.04
final_wd = 1e-6

# create model
model = MSN(vit)

device = torch.device("cuda:0")
model.to(device)

# custom transforms for CT task
MSNTransform = MSNTransform_CT()

# create a dataset from a folder containing images
dataset_train = LightlyDataset(path_to_data,
                         transform=MSNTransform)
def OriImg_loader(f):
    with open(f, "rb") as f:
        image = Image.open(f)
        return np.array(image)
dataset_train.dataset.loader = OriImg_loader

dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=16,
)

# loss
criterion = MSNLoss()

params = [
    *list(model.anchor_backbone.parameters()),
    *list(model.anchor_projection_head.parameters()),
    model.prototypes,
]
optimizer = torch.optim.AdamW(params, lr=start_lr)

# iterations_per_epoch
ipe = len(dataloader)
# schedulers
scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup_epochs*ipe),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(1.25*num_epochs*ipe))
wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_wd,
        T_max=int(1.25*num_epochs*ipe))

print("Starting Training")
since = time.time()

best_avg_loss = 1000
best_repStd = 0

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        views = batch[0]
        utils.update_momentum(model.anchor_backbone, model.backbone, 0.996)
        utils.update_momentum(
            model.anchor_projection_head, model.projection_head, 0.996
        )

        views = [view.to(device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        representations = model.backbone(targets)
        targets_out = model.projection_head(representations)
        anchors_out = model.forward_masked(anchors)
        anchors_focal_out = model.forward_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = criterion(anchors_out, targets_out, model.prototypes.data)
        total_loss += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        wd_scheduler.step()

    avg_loss = total_loss / len(dataloader)
    repStd = std_of_l2_normalized(representations)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.4f}, representation std:{repStd} ")

    if avg_loss < best_avg_loss and repStd > best_repStd:
        best_avg_loss = avg_loss
        best_repStd = repStd
        torch.save(model.state_dict(), results_save_folder +
                   'MSN_deepLesion_train_loss_{:.4f}_repStd_{:.4f}.pth'.format(avg_loss, repStd))
        print("model saved")
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
