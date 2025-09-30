from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms as T
import pickle
from msn_custom_ViT_model import MSN, vit
import cv2
import torch.nn.functional as F
from monai.utils import set_determinism
import os
import numpy as np
from scipy.stats import scoreatpercentile
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from FusionPhase_FM_ABMIL import Patches_ABMIL, FusionClassifier


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def apply_window(image, WL, WW):
    window_min = WL - WW / 2
    window_max = WL + WW / 2
    image_windowed = np.clip((image - window_min) / (window_max - window_min), 0, 1) * 255
    return image_windowed.astype(np.uint8)

def foreground_obtain(pca_features_bg):
    pca_bg_thresh = 0

    foreground = pca_features_bg > pca_bg_thresh

    foreground = np.squeeze(foreground)

    return foreground

def PCA_visualization(image, pca, pca_foreground, backbone):

    p_patch = T.ToTensor()(image)
    p_patch = p_patch.unsqueeze(0)
    p_patch = torch.as_tensor(p_patch, dtype=torch.float)
    p_patch = p_patch.to(device)

    patch_embeddings = backbone.transformer.encode(p_patch)
    patch_embeddings = patch_embeddings.cpu()
    patch_embeddings = patch_embeddings.squeeze(0)

    patch_embeddings = patch_embeddings[1:, :]
    # pca
    pca_features_all = pca.transform(patch_embeddings)

    # for some imag without foreground
    # 128X128 for better visualization
    resized_pca_nc_embeddings_array_zeros = np.zeros((128, 128, 3), dtype=np.uint8)
    resized_pca_nc_embeddings_img_flipped = Image.fromarray(resized_pca_nc_embeddings_array_zeros, mode='RGB')

    # remove the background
    pca_features_bg = pca_features_all[:, 0]

    foreground = foreground_obtain(pca_features_bg)

    if np.count_nonzero(foreground) != 0:

        pca_features = pca_foreground.transform(patch_embeddings[foreground])

        pca_features_1 = pca_features[:, 0]

        pca_features_2 = pca_features[:, 1]

        pca_features_3 = pca_features[:, 2]

        pca_features_foreground = np.stack(
            (pca_features_1, pca_features_2, pca_features_3), axis=-1)

        background = ~foreground
        pca_embeddings_img = pca_features_all
        pca_embeddings_img[background] = 0
        pca_embeddings_img[foreground] = pca_features_foreground

        pca_embeddings_img = pca_embeddings_img.reshape(16, 16, 3)

        pca_embeddings_img_norl = normalize_image(pca_embeddings_img)

        rgb_image_pil = Image.fromarray((pca_embeddings_img_norl * 255).astype(np.uint8), mode='RGB')

        resized_pca_nc_embeddings_img = rgb_image_pil.resize((64, 64), Image.LANCZOS)
        resized_pca_nc_embeddings_img_flipped = resized_pca_nc_embeddings_img.transpose(Image.FLIP_TOP_BOTTOM)

    patch_ori = image[:, :, 0]
    patch_ori = np.array(patch_ori, dtype=np.float32)
    patch_ori_CT = patch_ori * 3072 - 1024
  
    WL = 30
    WW = 300
    ct_image_windowed = apply_window(patch_ori_CT, WL, WW)
    ct_image_windowed_pil = Image.fromarray(ct_image_windowed.astype(np.uint8), mode='L')
    ct_image_windowed_pil_flipped = ct_image_windowed_pil.transpose(Image.FLIP_TOP_BOTTOM)

    # 128X128 for better visualization
    ct_image_windowed_pil_flipped = ct_image_windowed_pil_flipped.resize((128, 128), Image.LANCZOS)
    resized_pca_nc_embeddings_img_flipped = resized_pca_nc_embeddings_img_flipped.resize((128, 128), Image.LANCZOS)

    return ct_image_windowed_pil_flipped, resized_pca_nc_embeddings_img_flipped


def PCA_key_patches(patch_ori_list, backbone):

    patches_embeddings_all = []

    for patch_ori in patch_ori_list:
        p_patch = T.ToTensor()(patch_ori)
        p_patch = p_patch.unsqueeze(0)
        p_patch = torch.as_tensor(p_patch, dtype=torch.float)
        p_patch = p_patch.to(device)

        patch_embeddings = backbone.transformer.encode(p_patch)
        patch_embeddings = patch_embeddings.cpu()
        patch_embeddings = patch_embeddings.squeeze(0)

        # the embeddings of each patch in CT img
        patches_embeddings = patch_embeddings[1:, :]
        patches_embeddings_all.append(patches_embeddings)

    patches_embeddings_all = np.vstack(patches_embeddings_all)
    # Compute PCA between the patches of the image
    pca = PCA(n_components=3)
    pca.fit(patches_embeddings_all)

    # pca
    pca_features = pca.transform(patches_embeddings_all)

    # remove the background
    pca_features_bg = pca_features[:, 0]

    foreground = foreground_obtain(pca_features_bg)

    # Fit PCA foreground
    pca_foreground = PCA(n_components=3)
    pca_foreground.fit(patches_embeddings_all[foreground])

    return pca, pca_foreground


if __name__ == '__main__':

    set_determinism()

    # data folder
    data_folder_path = '/mnt/data/'

    # initialize the GPU
    device = torch.device("cuda:0")

    # # # model_path
    # # load trained model
    model_fm_fusion_path = '/models/LFM-fusion-model.pkl'

    print(model_fm_fusion_path)
    with open(model_fm_fusion_path, 'rb') as f:
        model_ft_dict = torch.load(f)
    # load model
    model = MSN(vit)
    num_classes_downstream = 2
    model_nc = Patches_ABMIL(model, 512)
    model_ap = Patches_ABMIL(model, 512)
    model_pv = Patches_ABMIL(model, 512)
    model_ft = FusionClassifier(model_nc, model_ap, model_pv, 512, num_classes_downstream)

    model_ft.load_state_dict(model_ft_dict)
    model_ft.to(device)
    model_ft.eval()
    torch.set_grad_enabled(False)

    outputs_p_list_MVI = []
    outputs_p_list_PathologicalGrade = []

    # load data
    all_PID = pickle.load(open(data_folder_path + '/HCC_ID.bin', "rb"))

    p_patches_nc_list = pickle.load(
        open(data_folder_path + '/HCC_patches_non-contrast.bin', "rb"))
    p_patches_ap_list = pickle.load(
        open(data_folder_path + '/HCC_patches_arterial-phase.bin', "rb"))
    p_patches_pv_list = pickle.load(
        open(data_folder_path + '/HCC_patches_portal-venous.bin', "rb"))
    p_patches_mask_list = pickle.load(
        open(data_folder_path + '/HCC_patches_mask.bin', "rb"))

    
    # find the key patches according to the weights from MIL attention
    p_key_patches_nc = []
    p_key_patches_ap = []
    p_key_patches_pv = []

    p_key_patches_nc_mask = []
    p_key_patches_ap_mask = []
    p_key_patches_pv_mask = []

    for p_id in range(0, len(p_patches_nc_list)):
        p_patches_nc = p_patches_nc_list[p_id]
        p_patches_ap = p_patches_ap_list[p_id]
        p_patches_pv = p_patches_pv_list[p_id]
        p_patches_mask = p_patches_mask_list[p_id]

        patch_num = len(p_patches_nc)

        p_pred, A_nc, A_ap, A_pv = model_ft(p_patches_nc, p_patches_ap, p_patches_pv)

        # save predictions
        p_pred = p_pred.cpu()
        p_pred = p_pred.detach().numpy()

        p_pred_MVI = p_pred[0]
        p_pred_PathologicalGrade = p_pred[1]

        outputs_p_list_MVI.append(p_pred_MVI)
        outputs_p_list_PathologicalGrade.append(p_pred_PathologicalGrade)

        # find the key patches
        A_nc = A_nc.cpu()
        A_nc = A_nc.detach().numpy()
        nc_max_index = np.argmax(A_nc)

        A_ap = A_ap.cpu()
        A_ap = A_ap.detach().numpy()
        ap_max_index = np.argmax(A_ap)

        A_pv = A_pv.cpu()
        A_pv = A_pv.detach().numpy()
        pv_max_index = np.argmax(A_pv)

        # collect the key patches
        p_key_patches_nc.append(p_patches_nc[nc_max_index])
        p_key_patches_ap.append(p_patches_ap[ap_max_index])
        p_key_patches_pv.append(p_patches_pv[pv_max_index])

        p_key_patches_nc_mask.append(p_patches_mask[nc_max_index])
        p_key_patches_ap_mask.append(p_patches_mask[ap_max_index])
        p_key_patches_pv_mask.append(p_patches_mask[pv_max_index])

    # use the corresponding backbone
    backbone_nc = model_ft.branch_nc
    backbone_ap = model_ft.branch_ap
    backbone_pv = model_ft.branch_pv

    # load pca
    with open(data_folder_path + '/pca_data_exBG.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    PCA_nc = loaded_data['PCA_nc']
    PCA_nc_foreground = loaded_data['PCA_nc_foreground']

    PCA_ap = loaded_data['PCA_ap']
    PCA_ap_foreground = loaded_data['PCA_ap_foreground']

    PCA_pv = loaded_data['PCA_pv']
    PCA_pv_foreground = loaded_data['PCA_pv_foreground']

    # save the CT patch and corresponding PCA color map for each patient
    for index in range(0, len(p_key_patches_nc)):
        p_id = all_PID[index]
        p_key_patch_nc = p_key_patches_nc[index]
        p_key_patch_ap = p_key_patches_ap[index]
        p_key_patch_pv = p_key_patches_pv[index]

        CT_ori_nc, PCA_map_nc = PCA_visualization(p_key_patch_nc, PCA_nc, PCA_nc_foreground, backbone_nc)
        CT_ori_ap, PCA_map_ap = PCA_visualization(p_key_patch_ap, PCA_ap, PCA_ap_foreground, backbone_ap)
        CT_ori_pv, PCA_map_pv = PCA_visualization(p_key_patch_pv, PCA_pv, PCA_pv_foreground, backbone_pv)

        p_key_patch_nc_mask = p_key_patches_nc_mask[index]
        p_key_patch_ap_mask = p_key_patches_ap_mask[index]
        p_key_patch_pv_mask = p_key_patches_pv_mask[index]
        
        p_key_patch_nc_mask = np.flipud(p_key_patch_nc_mask)
        p_key_patch_ap_mask = np.flipud(p_key_patch_ap_mask)
        p_key_patch_pv_mask = np.flipud(p_key_patch_pv_mask)

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        axes[0, 0].imshow(CT_ori_nc, cmap='gray')
        axes[0, 0].set_title('CT_ori_nc', fontsize=14, fontweight='bold')
        axes[0, 1].imshow(PCA_map_nc)
        axes[0, 1].set_title('PCA_map_nc', fontsize=14, fontweight='bold')
        axes[0, 2].imshow(p_key_patch_nc_mask)
        axes[0, 2].set_title('mask', fontsize=14, fontweight='bold')

        axes[1, 0].imshow(CT_ori_ap, cmap='gray')
        axes[1, 0].set_title('CT_ori_ap', fontsize=14, fontweight='bold')
        axes[1, 1].imshow(PCA_map_ap)
        axes[1, 1].set_title('PCA_map_ap', fontsize=14, fontweight='bold')
        axes[1, 2].imshow(p_key_patch_ap_mask)
        axes[1, 2].set_title('mask', fontsize=14, fontweight='bold')

        axes[2, 0].imshow(CT_ori_pv, cmap='gray')
        axes[2, 0].set_title('CT_ori_pp', fontsize=14, fontweight='bold')
        axes[2, 1].imshow(PCA_map_pv)
        axes[2, 1].set_title('PCA_map_pp', fontsize=14, fontweight='bold')
        axes[2, 2].imshow(p_key_patch_pv_mask)
        axes[2, 2].set_title('mask', fontsize=14, fontweight='bold')

        fig.text(0.5, 0.02, f'ID = {p_id}', ha='center', fontsize=18, fontweight='bold')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        output_jpgs_folder = '/data/PCA_visualization/'

        if not os.path.exists(output_jpgs_folder):
            os.makedirs(output_jpgs_folder)

        output_file_path = output_jpgs_folder + f'{p_id}.jpg'
        plt.savefig(output_file_path, dpi=300)





