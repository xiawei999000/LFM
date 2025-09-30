import copy
import torchvision
from torch import nn
from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTorchvision
from lightly.models.modules.heads import MSNProjectionHead

mask_ratio = 0.3
image_size = 64
patch_size = 4

feature_num = 512
num_layers = 6
num_heads = 4


class MSN(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.projection_head = MSNProjectionHead(input_dim=feature_num)  # feature num

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight  # no change

    def forward(self, images):
        out = self.backbone(images=images)
        return self.projection_head(out)

    def forward_masked(self, images):
        batch_size, _, _, width = images.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        out = self.anchor_backbone(images=images, idx_keep=idx_keep)
        return self.anchor_projection_head(out)

# ViT small configuration
vit = torchvision.models.VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=feature_num,
    mlp_dim=feature_num * 4,
)