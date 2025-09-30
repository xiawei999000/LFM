from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as T
from PIL.Image import Image
import numpy as np
from torch import Tensor
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
import numpy as np
from monai import transforms as monai_transforms

CT_NORMALIZE = {"mean": [31744], "std": [3072]}
class CTNormalize:
    def __init__(self, CT_mean: float = 31744.0, CT_std: float = 3072.0):
        self.CT_mean = CT_mean
        self.CT_std = CT_std
    def __call__(self, image: np.ndarray) -> np.ndarray:
        normalized_image = (image - self.CT_mean) / self.CT_std
        image_3channels = np.stack([normalized_image, normalized_image, normalized_image], axis=-1)
        return image_3channels.astype(np.float32)

class MSNTransform_CT(MultiViewTransform):
    def __init__(
        self,
        random_size: int = 64,
        focal_size: int = 32,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.7, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.5, 0.8),
        gaussian_blur: float = 0.5,
        sigmas: Tuple[float, float] = (0.1, 1),
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
    ):
        random_view_transform = MSNViewTransform_CT(
            crop_size=random_size,
            crop_scale=random_crop_scale,
            gaussian_blur=gaussian_blur,
            sigmas=sigmas,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
        )
        focal_view_transform = MSNViewTransform_CT(
            crop_size=focal_size,
            crop_scale=focal_crop_scale,
            gaussian_blur=gaussian_blur,
            sigmas=sigmas,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
        )
        transforms = [random_view_transform] * random_views
        transforms += [focal_view_transform] * focal_views
        super().__init__(transforms=transforms)


class MSNViewTransform_CT:
    def __init__(
        self,
        crop_size: int = 64,
        crop_scale: Tuple[float, float] = (0.3, 1.0),
        gaussian_blur: float = 0.5,
        sigmas: Tuple[float, float] = (0.1, 2),
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
    ):
        transform = [
            CTNormalize(),
            T.ToTensor(),
            T.RandomResizedCrop(size=crop_size, scale=crop_scale),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            monai_transforms.RandGaussianSmooth(prob=gaussian_blur, sigma_x=(sigmas[0], sigmas[1]),
                                                sigma_y=(sigmas[0], sigmas[1])),
        ]

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed
