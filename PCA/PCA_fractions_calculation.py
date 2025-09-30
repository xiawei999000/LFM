
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
import pandas as pd
from scipy.stats import scoreatpercentile
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from FusionPhase_FM_ABMIL import Patches_ABMIL, FusionClassifier


def foreground_obtain(pca_features_bg):
    pca_bg_thresh = 0
    foreground = pca_features_bg > pca_bg_thresh
    foreground = np.squeeze(foreground)
    return foreground

def normalize_image(image):

    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

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


def PCA_component_counts(image, pca, pca_foreground, backbone):
    p_patch = T.ToTensor()(image)
    p_patch = p_patch.unsqueeze(0)
    p_patch = torch.as_tensor(p_patch, dtype=torch.float)
    p_patch = p_patch.to(device)

    patch_embeddings = backbone.transformer.encode(p_patch)
    patch_embeddings = patch_embeddings.cpu()
    patch_embeddings = patch_embeddings.squeeze(0)

    patch_embeddings = patch_embeddings[1:, :]

    # pca
    pca_features = pca.transform(patch_embeddings)

    # remove the background
    pca_features_bg = pca_features[:, 0]

    foreground = foreground_obtain(pca_features_bg)

    patch_embeddings_foreground = patch_embeddings[foreground]

    channel_counts = np.zeros(3)

    if len(patch_embeddings_foreground) != 0:
        pca_features = pca_foreground.transform(patch_embeddings_foreground)
        max_channel_indices = np.argmax(pca_features, axis=1)
        channel_counts = np.bincount(max_channel_indices.flatten(), minlength=3)

    return channel_counts

if __name__ == '__main__':

    set_determinism()

    data_folder_path = '/data/HCC-bin/train/'

    # initialize the GPU
    device = torch.device("cuda:0")

    # load data
    label_obj_list = ['MVI', 'PathologicalGrade']

    labels_dict = {}
    for label_name in label_obj_list:
        label_path = os.path.join(data_folder_path, 'HCC_' + label_name + '.bin')

        labels_dict[label_name] = pickle.load(open(label_path, "rb"))
    labels_array = np.column_stack([labels_dict[label] for label in labels_dict])

    all_PID = pickle.load(open(data_folder_path + '/HCC_ID.bin', "rb"))
    all_MVI = pickle.load(open(data_folder_path + '/HCC_MVI.bin', "rb"))
    all_PathologicalGrade = pickle.load(open(data_folder_path + '/HCC_PathologicalGrade.bin', "rb"))

    columns = ['ID', 'MVI', 'PathologicalGrade',
               'model_fm_fusion_preds_MVI', 'model_fm_fusion_preds_PathologicalGrade',
               'nc_pca_1_fraction', 'nc_pca_2_fraction', 'nc_pca_3_fraction',
               'ap_pca_1_fraction', 'ap_pca_2_fraction', 'ap_pca_3_fraction',
               'pv_pca_1_fraction', 'pv_pca_2_fraction', 'pv_pca_3_fraction'
               ]

    results = pd.DataFrame(columns=columns)
    results = results.assign(ID=all_PID)
    results = results.assign(MVI=all_MVI)
    results = results.assign(PathologicalGrade=all_PathologicalGrade)

    # # # model_path
    # # load trained model
    model_fm_fusion_path = './models/LFM-fusion-model.pkl'
    print(model_fm_fusion_path)
    with open(model_fm_fusion_path, 'rb') as f:
        model_ft_dict = torch.load(f)

    model = MSN(vit)
    num_classes_downstream = len(label_obj_list)
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
    p_patches_nc_list = pickle.load(
        open(data_folder_path + '/HCC_patches_non-contrast.bin', "rb"))
    p_patches_ap_list = pickle.load(
        open(data_folder_path + '/HCC_patches_arterial-phase.bin', "rb"))
    p_patches_pv_list = pickle.load(
        open(data_folder_path + '/HCC_patches_portal-venous.bin', "rb"))

    # find the key patches according to the weights from MIL attention
    p_key_patches_nc = []
    p_key_patches_ap = []
    p_key_patches_pv = []

    for p_id in range(0, len(p_patches_nc_list)):
        p_patches_nc = p_patches_nc_list[p_id]
        p_patches_ap = p_patches_ap_list[p_id]
        p_patches_pv = p_patches_pv_list[p_id]

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

    # pca
    # use the corresponding backbone
    backbone_nc = model_ft.branch_nc
    backbone_ap = model_ft.branch_ap
    backbone_pv = model_ft.branch_pv

    # fit pca
    PCA_nc, PCA_nc_foreground = PCA_key_patches(p_key_patches_nc, backbone_nc)
    PCA_ap, PCA_ap_foreground = PCA_key_patches(p_key_patches_ap, backbone_ap)
    PCA_pv, PCA_pv_foreground = PCA_key_patches(p_key_patches_pv, backbone_pv)

    # store fractions
    component_fractions_nc_1_list = []
    component_fractions_nc_2_list = []
    component_fractions_nc_3_list = []

    component_fractions_ap_1_list = []
    component_fractions_ap_2_list = []
    component_fractions_ap_3_list = []

    component_fractions_pv_1_list = []
    component_fractions_pv_2_list = []
    component_fractions_pv_3_list = []

    # calculate component fractions for each patient
    for p_id in range(0, len(p_patches_nc_list)):
        p_key_patch_nc = p_key_patches_nc[p_id]
        p_key_patch_ap = p_key_patches_ap[p_id]
        p_key_patch_pv = p_key_patches_pv[p_id]

        component_counts_nc = np.zeros(3)
        component_counts_ap = np.zeros(3)
        component_counts_pv = np.zeros(3)

        component_counts_nc = PCA_component_counts(p_key_patch_nc, PCA_nc, PCA_nc_foreground, backbone_nc)
        component_counts_ap = PCA_component_counts(p_key_patch_ap, PCA_ap, PCA_ap_foreground, backbone_ap)
        component_counts_pv = PCA_component_counts(p_key_patch_pv, PCA_pv, PCA_pv_foreground, backbone_pv)

        component_fractions_nc = np.zeros(3)
        component_fractions_ap = np.zeros(3)
        component_fractions_pv = np.zeros(3)

        # fractions for each person
        if np.sum(component_counts_nc) != 0:
            component_fractions_nc = component_counts_nc / np.sum(component_counts_nc)
        if np.sum(component_counts_ap) != 0:
            component_fractions_ap = component_counts_ap / np.sum(component_counts_ap)
        if np.sum(component_counts_pv) != 0:
            component_fractions_pv = component_counts_pv / np.sum(component_counts_pv)

        # store fractions
        component_fractions_nc_1_list.append(component_fractions_nc[0])
        component_fractions_nc_2_list.append(component_fractions_nc[1])
        component_fractions_nc_3_list.append(component_fractions_nc[2])

        component_fractions_ap_1_list.append(component_fractions_ap[0])
        component_fractions_ap_2_list.append(component_fractions_ap[1])
        component_fractions_ap_3_list.append(component_fractions_ap[2])

        component_fractions_pv_1_list.append(component_fractions_pv[0])
        component_fractions_pv_2_list.append(component_fractions_pv[1])
        component_fractions_pv_3_list.append(component_fractions_pv[2])


    results = results.assign(model_fm_fusion_preds_MVI=outputs_p_list_MVI)
    results = results.assign(model_fm_fusion_preds_PathologicalGrade=outputs_p_list_PathologicalGrade)

    results = results.assign(nc_pca_1_fraction=component_fractions_nc_1_list)
    results = results.assign(nc_pca_2_fraction=component_fractions_nc_2_list)
    results = results.assign(nc_pca_3_fraction=component_fractions_nc_3_list)

    results = results.assign(ap_pca_1_fraction=component_fractions_ap_1_list)
    results = results.assign(ap_pca_2_fraction=component_fractions_ap_2_list)
    results = results.assign(ap_pca_3_fraction=component_fractions_ap_3_list)

    results = results.assign(pv_pca_1_fraction=component_fractions_pv_1_list)
    results = results.assign(pv_pca_2_fraction=component_fractions_pv_2_list)
    results = results.assign(pv_pca_3_fraction=component_fractions_pv_3_list)

    output_file = data_folder_path + '/pca-fracs-values.xlsx'
    results.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


