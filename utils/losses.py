# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import numpy as np

class DiceCELoss(nn.Module):
    """
    Fonction de perte combinant la Cross-Entropy et le Dice Score.
    Particulièrement adaptée pour les déséquilibres de classes en segmentation.
    """
    def __init__(self, weight=None, ignore_index=None, dice_weight=1.0, ce_weight=1.0, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        if ignore_index is not None:
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=logits.shape[1])
        target_onehot = target_onehot.permute(0, 2, 1).float()
        
        dims = (0, 2)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1. - dice_score.mean()
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

def get_loss(cfg, weight=None):
    """
    Instancie et retourne la fonction de perte spécifiée dans la configuration.
    Gère le parsing des poids selon leur format d'entrée (ListConfig, list, str).
    """
    if weight is not None:
        
        # Traitement des pondérations fournies sous forme de structure list/ListConfig
        if isinstance(weight, (ListConfig, list)):
            
            # Cas spécifique : Parsing requis si la liste contient des objets String
            if len(weight) > 0 and isinstance(weight[0], str):
                full_str = " ".join(weight)
                clean_str = full_str.replace('[', '').replace(']', '').replace('\n', ' ')
                weight_np = np.fromstring(clean_str, sep=' ')
                weight = torch.tensor(weight_np).float()
            
            # Cas standard : Liste de valeurs numériques
            else:
                weight = torch.tensor(weight).float()

        # Traitement des pondérations fournies sous forme de chaîne de caractères unique
        elif isinstance(weight, str):
            clean_str = weight.replace('[', '').replace(']', '').replace('\n', ' ')
            weight_np = np.fromstring(clean_str, sep=' ')
            weight = torch.tensor(weight_np).float()
    
    if cfg.loss.type == "ce":
        return nn.CrossEntropyLoss(weight=weight, ignore_index=cfg.loss.ignore_index)
    elif cfg.loss.type == "dice_ce":
        return DiceCELoss(
            weight=weight, 
            ignore_index=cfg.loss.ignore_index,
            dice_weight=cfg.loss.dice_weight,
            ce_weight=cfg.loss.ce_weight
        )
    else:
        raise ValueError(f"Loss type {cfg.loss.type} inconnue")