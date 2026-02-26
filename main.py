# main.py
import hydra
from omegaconf import DictConfig, OmegaConf , open_dict
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict

from data.dataset import SulcalDataset
from training.augmentations import transform
from models.factory import get_model
from utils.losses import get_loss
from training.trainer import Trainer
from evaluation.eval import evaluate  # Assure-toi que ta fonction evaluate retourne un dict


def collate_fn(batch):
    coords_list, faces_list, X_list, Y_list = zip(*batch)
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)
    return coords_list[0], faces_list[0], X, Y

def get_patient_indices(dataset, patient_list):
    indices = []
    for idx, sample in enumerate(dataset.samples):
        if sample['patient'] in patient_list:
            indices.append(idx)
    return indices

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Configuration initiale et Seed
    torch.manual_seed(cfg.cross_validation.seed)
    np.random.seed(cfg.cross_validation.seed)
    
    if 'augmentations' in cfg and 'augmentations' not in cfg.data:
         cfg.data.augmentations = cfg.augmentations

    # Initialisation du dataset complet
    full_dataset = SulcalDataset(
        cfg=cfg.data,  
        base_dir=cfg.paths.base_dir,     
        transform=transform 
    )

    # Séparation des patients (Train/Val vs Test)
    all_patients = sorted(list(set([s['patient'] for s in full_dataset.samples])))
    
    # Isolation du Test Set
    cv_patients, test_patients = train_test_split(
        all_patients, 
        test_size=cfg.cross_validation.test_ratio, 
        random_state=cfg.cross_validation.seed,
        shuffle=True
    )

    # Sauvegarde des identifiants du Test Set
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.output_dir, "test_patients_list.txt"), "w") as f:
        for p in test_patients:
            f.write(f"{p}\n")
            
    print(f"📊 Total Patients: {len(all_patients)}")
    print(f"🔒 Test Set: {len(test_patients)} patients mis de côté (voir test_patients_list.txt)")
    print(f"🔄 Cross-Validation Pool: {len(cv_patients)} patients")

    # Suivi des métriques de validation
    val_scores = defaultdict(list)

    # Configuration des splits (Cross-Validation ou Holdout)
    k = cfg.cross_validation.k_folds
    splits = []

    indices = np.arange(len(cv_patients))

    if k > 1:
        print(f"🔄 Mode Cross-Validation activé ({k} folds)")
        kf = KFold(n_splits=k, shuffle=True, random_state=cfg.cross_validation.seed)
        splits = list(kf.split(cv_patients))
    else:
        print(f"⚠️ Mode Run Unique (k_folds=1)")
        print(f"   -> Simple séparation Train/Val (65%/35%) sur le pool restant.")
        
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=0.35, 
            random_state=cfg.cross_validation.seed,
            shuffle=True
        )
        splits = [(train_idx, val_idx)]
        
    all_patients_metrics = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*40}")
        if k > 1:
            print(f"🚀 FOLD {fold+1}/{k}")
        else:
            print(f"🚀 SINGLE RUN (Train/Val Split)")
        print(f"{'='*40}")

        fold_train_patients = [cv_patients[i] for i in train_idx]
        fold_val_patients = [cv_patients[i] for i in val_idx]
        # print(f"\n📝 [Fold {fold+1}] Sujets de VALIDATION ({len(fold_val_patients)}) :")
        # print(fold_val_patients)

        # Datasets & Loaders
        train_ds = Subset(full_dataset, get_patient_indices(full_dataset, fold_train_patients))
        val_ds = Subset(full_dataset, get_patient_indices(full_dataset, fold_val_patients))

        train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True, 
                                  collate_fn=collate_fn, num_workers=cfg.data.num_workers)
        val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, shuffle=False, 
                                collate_fn=collate_fn, num_workers=cfg.data.num_workers)

        # Initialisation du modèle
        device = torch.device(cfg.device)
        model = get_model(cfg, device)

        base_lr = cfg.training.lr
        # Définition du Learning Rate (fallback sur base_lr si encoder_lr est absent)
        encoder_lr = cfg.training.get('encoder_lr', base_lr)

        params_to_optimize = []

        # Gestion du Fine-Tuning (LR différentiel pour le backbone)
        if hasattr(model, 'backbone') and encoder_lr != base_lr:
            print(f"⚡ Mode Fine-Tuning activé : Encoder LR={encoder_lr} | Decoder LR={base_lr}")
            
            backbone_ids = list(map(id, model.backbone.parameters()))
            
            # Groupe de paramètres 1 : Backbone
            params_to_optimize.append({
                'params': model.backbone.parameters(), 
                'lr': encoder_lr
            })
            
            # Groupe de paramètres 2 : Reste du modèle
            rest_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())
            params_to_optimize.append({
                'params': rest_params, 
                'lr': base_lr
            })
            
        else:
            # Mode d'entraînement standard (LR unique)
            print(f"⚡ Mode Standard : LR unique = {base_lr} pour tout le modèle.")
            params_to_optimize = model.parameters()

        # Initialisation de l'optimiseur
        optimizer = torch.optim.AdamW(
            params_to_optimize, 
            lr=base_lr, 
            weight_decay=cfg.training.weight_decay
        )

        if cfg.loss.weight ==-1: 
            criterion = get_loss(cfg).to(device)
        else : 
            criterion = get_loss(cfg,cfg.loss.weight).to(device)

        if cfg.cross_validation.k_folds > 1:
            fold_save=True
        else : 
            fold_save=False 
            
        start_epoch = 0
        best_iou = 0.0
        
        # Restauration depuis un checkpoint si spécifié
        resume_path = cfg.get("resume_path", None) 

        if resume_path and os.path.exists(resume_path):
            print(f"♻️  Chargement du checkpoint : {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device,weights_only=False)
            
            # Chargement des poids du modèle
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restauration de l'état de l'optimiseur
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Mise à jour du LR de l'optimiseur avec la config courante
                current_lr = cfg.training.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"   -> Optimizer chargé mais LR forcé à : {current_lr}")

            # Récupération de l'époque de reprise et des métriques
            start_epoch = checkpoint.get('epoch', 0) + 1 
            best_iou = checkpoint.get('best_iou', 0.0)
            
            print(f"   -> Reprise à l'epoch {start_epoch} avec Best IoU précédent : {best_iou:.4f}")

        # Initialisation de l'entraîneur
        trainer = Trainer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_save=fold_save,
            fold=fold,
            start_epoch=start_epoch, 
            best_iou=best_iou        
        )

        # Lancement de l'entraînement
        trainer.run() 

        # --- ÉVALUATION DES PERFORMANCES DU FOLD ---
        print(f"✅ Calcul des scores de validation pour le Fold {fold+1}...")
        
        if fold_save: 
            fold_output_dir=os.path.join(cfg.paths.output_dir, f"fold_{fold}")
        else : 
            fold_output_dir=cfg.paths.output_dir

        # Chargement du meilleur modèle du fold courant
        best_model_path = os.path.join(fold_output_dir, "best_model.pth")
        
        if  os.path.exists(best_model_path) : 
            checkpoint = torch.load(best_model_path, map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # Fallback sur le modèle initial si aucune amélioration n'a eu lieu
        elif not os.path.exists(best_model_path) and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

        # Évaluation sur le set de validation
        metrics = evaluate(model, val_loader, cfg, device, epoch=999, plot=False)
    
        all_patients_metrics['iou'].extend(metrics['per_subject_iou'])
        all_patients_metrics['dice'].extend(metrics['per_subject_dice'])
        all_patients_metrics['esi'].extend(metrics['per_subject_esi'])
        all_patients_metrics['invesi'].extend(metrics['per_subject_invesi'])
        #all_patients_metrics['elocal'].extend(metrics['per_subject_elocal'])
        all_patients_metrics['elocal_matrices'].append(metrics['elocal_matrix'])

    print(f"\n{'='*60}")
    print(f"📊 RÉSULTATS FINAUX SUR {len(all_patients_metrics['iou'])} SUJETS")
    print(f"{'='*60}")
    
    val_indices = val_ds.indices
    patient_names = [full_dataset.samples[idx]['patient'] for idx in val_indices]
    
    # Mapping des scores bruts par patient
    raw_data_fold = {
        name: scores.tolist() 
        for name, scores in zip(patient_names, metrics['elocal_matrix'])
    }

    print(f"\n--- RAW DATA FOLD {fold} ---")
    #print(raw_data_fold)
    
    for metric_name in ['iou', 'dice', 'esi','invesi']:
        scores = np.array(all_patients_metrics[metric_name])
        print(f"\n🔹 {metric_name.upper()}:")
        print(f"   Moyenne globale : {scores.mean():.4f}")
        print(f"   Écart-type (Std): {scores.std():.4f}")
        print(f"   Min / Max       : {scores.min():.4f} / {scores.max():.4f}")

    print(f"\n{'='*60}")
    print(f"📊 RÉSULTATS FINAUX PAR SILLON (LOGIQUE AGGRÉGÉE)")
    print(f"{'='*60}")
    
    all_patients_elocal_matrices=all_patients_metrics['elocal_matrices']
    
    # Concaténation des matrices ELOCAL : (Total_Sujets, Num_Classes)
    if len(all_patients_elocal_matrices) > 0:
        full_matrix = np.vstack(all_patients_elocal_matrices) 
        
        # Calcul de la moyenne par classe (sillon) en ignorant les valeurs manquantes (NaN)
        mean_per_sulcus = np.nanmean(full_matrix, axis=0)
        
        # Filtrage des classes n'ayant aucune donnée valide
        valid_sulci_mask = ~np.isnan(mean_per_sulcus)
        valid_scores = mean_per_sulcus[valid_sulci_mask]

        print(f"🔹 ELOCAL (Stats sur les moyennes par sillon) :")
        print(f"   Nombre de sillons valides : {len(valid_scores)}")
        print(f"   Moyenne globale           : {np.mean(valid_scores):.4f}")
        print(f"   Écart-type (entre sillons): {np.std(valid_scores):.4f}")
        print(f"   Min (Meilleur sillon)     : {np.min(valid_scores):.4f}")
        print(f"   Max (Pire sillon)         : {np.max(valid_scores):.4f}")
        
        for i, score in enumerate(mean_per_sulcus):
            if not np.isnan(score):
                print(f"   Sillon {i}: {score:.4f}")

    else:
        print("Aucune donnée collectée.")

if __name__ == "__main__":
    main()