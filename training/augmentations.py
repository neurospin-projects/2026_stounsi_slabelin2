from surfify.augmentation import (
SurfCutOut, SurfNoise, SurfBlur, SurfRotation, HemiMixUp, GroupMixUp,
Transformer, interval)
import torch 
from surfify.utils import icosahedron , number_of_ico_vertices , neighbors
from surfify.utils.sampling import rotate_data
from random import random
import numpy as np
import os 

"""
def transform(coords, faces, mask_output,mask_transfer,neighs,angles=None):
    print ("starting augmentation")
    if random() > 0.1:

        #aug = SurfRotation(coords, faces,  phi=interval((5, 180), float), theta=0,psi=0, cachedir="/neurospin/dico/stounsi/Runs/spherical_labelling/data/surfify_cache_augmentation" ,interpolation='euclidian' )
        
        #mask_output=aug(mask_output.cpu().numpy())
        mask_output=mask_output.view(1,163842,1).cpu().numpy()
        if angles is None:
            #generate a3 uplet , which is from normal distriubtion mean 0 and std pi/4
            angles = np.random.normal(0, np.pi/4, 3)

        mask_output=rotate_data(mask_output, coords.cpu().numpy(), faces.cpu().numpy(),angles ,interpolation='barycentric', neighs=neighs, weights=None)
        mask_transfer=rotate_data(mask_transfer.view(1,163842,1).cpu().numpy(), coords.cpu().numpy(), faces.cpu().numpy(),angles ,interpolation='barycentric', neighs=neighs, weights=None)
        
        print ("end augmentation")
        mask_output=torch.from_numpy(mask_output)
        mask_transfer=torch.from_numpy(mask_transfer)



        return coords, faces, mask_output,mask_transfer
    else:
        return coords, faces, mask_output,mask_transfer"""
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from random import random

def transform(coords, faces, inputs, targets, cfg=None):
    """
    Args:
        inputs: Tensor (C, N) - Canal 0 = Lignes, Canaux 1+ = Features.
        targets: Tensor (N) - Labels.
        cfg: Configuration Hydra.
    """
    
    # Paramétrage des probabilités et intensités d'augmentation
    if cfg:
        prob_geo = cfg.augmentations.prob_geometric
        prob_int = cfg.augmentations.prob_intensity
        noise_sigma = cfg.augmentations.noise_sigma
        scale_limit = cfg.augmentations.scale_limit
        rotate_angle_std=cfg.augmentations.rotate_angle_std
    else:
        prob_geo, prob_int = 0.5, 0.5
        noise_sigma, scale_limit = 0.05, 0.1

    # --- Augmentations d'intensité (Exclusion stricte du canal 0) ---
    if random() < prob_int and inputs.shape[0] > 1:
        
        features = inputs[1:].clone()
        
        # Bruit Gaussien
        if random() < 0.5:
            noise = torch.randn_like(features) * noise_sigma
            features = features + noise
            
        # Ajustement du contraste (Scaling)
        if random() < 0.5:
            scale = 1.0 + (torch.rand(1) * 2 * scale_limit - scale_limit)
            features = features * scale
            
        inputs[1:] = features

    # --- Augmentations géométriques (Rotations 3D) ---
    if random() < prob_geo:
        inputs_np = inputs.numpy()
        targets_np = targets.numpy()
        coords_np = coords.numpy()

        # Matrice de rotation
        angles = np.random.normal(0, rotate_angle_std, 3) 
        Rmat = R.from_euler('ZYX', angles, degrees=False)
        rotated_coords = Rmat.apply(coords_np)

        # Structure spatiale pour l'interpolation
        tree = cKDTree(coords_np)
        
        # Interpolation Nearest Neighbor (Préserve l'intégrité du canal 0 et des labels)
        _, idx_nn = tree.query(rotated_coords, k=1)
        
        targets_new = targets_np[idx_nn]
        lines_new = inputs_np[0][idx_nn] 

        # Interpolation pseudo-linéaire (Moyenne pondérée spatiale pour les features)
        if inputs_np.shape[0] > 1:
            dists, idx_k = tree.query(rotated_coords, k=3)
            
            dists = np.maximum(dists, 1e-10)
            weights = 1.0 / dists
            weights = weights / np.sum(weights, axis=1, keepdims=True) 
            
            features_new = []
            for c in range(1, inputs_np.shape[0]):
                feat_vals = inputs_np[c][idx_k] 
                interpolated = np.sum(feat_vals * weights, axis=1)
                features_new.append(interpolated)
            
            features_new = np.stack(features_new, axis=0)
            
            # Assemblage : Lignes (NN) + Features (Interpolées)
            inputs_new = np.concatenate([lines_new[None, :], features_new], axis=0)
        else:
            inputs_new = lines_new[None, :]

        inputs = torch.from_numpy(inputs_new).float()
        targets = torch.from_numpy(targets_new).long()

    return coords, faces, inputs, targets