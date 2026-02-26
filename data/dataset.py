import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from surfify.utils import icosahedron
from omegaconf import ListConfig

class SulcalDataset(Dataset):
    def __init__(self, base_dir, cfg, transform=None):
        """
        Args:
            base_dir (str): Chemin racine contenant les dossiers patients.
            cfg (DictConfig): Configuration du dataset.
            transform (callable, optional): Fonction d'augmentation des données.
        """
        self.cfg = cfg
        self.root_dir = base_dir
        
        self.hemi = cfg.hemisphere
        self.order = cfg.ico_order
        self.nb_vertex_thousands=cfg.nb_vertex_thousands
        self.transform = transform
        
        self.input_patterns = cfg.inputs
        self.target_pattern = cfg.target
        
        self.coords, self.faces = icosahedron(order=self.order, standard_ico=True)
        self.coords = torch.from_numpy(self.coords).float()
        self.faces = torch.from_numpy(self.faces).long()

        self.map_to_unknown = False
        raw_param = cfg.get('sulci_to_unknown', -1)

        if isinstance(raw_param, (list, ListConfig, tuple)):
            print(f"🧹 Filtrage activé : Les sillons {raw_param} deviendront 'Unknown' (2)")
            
            # Conversion pour un masquage optimisé dans __getitem__
            self.labels_to_replace = torch.tensor(list(set(raw_param)), dtype=torch.long)
            self.map_to_unknown = True
            
        elif raw_param != -1:
             print(f"⚠️ Paramètre 'sulci_to_unknown' ignoré ou mal formé : {raw_param}")

        self.samples = self._load_samples()

    def _load_samples(self):
        valid_samples = []
        
        if not os.path.exists(self.root_dir):
             print(f"Attention: {self.root_dir} introuvable.")
             return []

        patients = sorted([p for p in os.listdir(self.root_dir) 
                           if os.path.isdir(os.path.join(self.root_dir, p))])

        print(f"🔍 Scan des données ({self.hemi}, ordre {self.order})...")

        for patient in patients:
            surf_dir = os.path.join(self.root_dir, patient, "surf")
            
            format_args = {"hemi": self.hemi, "order": self.order, "nb_vertex_thousands" : self.nb_vertex_thousands }
            
            input_paths = []
            missing_input = False
            for pattern in self.input_patterns:
                fname = pattern.format(**format_args)
                fpath = os.path.join(surf_dir, fname)
                if not os.path.exists(fpath):
                    missing_input = True
                    break
                input_paths.append(fpath)
            
            if missing_input:
                continue

            fname_target = self.target_pattern.format(**format_args)
            target_path = os.path.join(surf_dir, fname_target)
            
            if os.path.exists(target_path):
                valid_samples.append({
                    "patient": patient,
                    "inputs": input_paths,
                    "target": target_path
                })
        
        print(f"✅ {len(valid_samples)} patients valides trouvés.")
        return valid_samples

    def _load_morph_data(self, path):
        if path.endswith('.gii'):
            data = nib.load(path).darrays[0].data
        else:
            data = nib.freesurfer.read_morph_data(path)
        return data.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Inputs (Shape: C, N)
        input_tensors = [torch.from_numpy(self._load_morph_data(p)) for p in sample["inputs"]]
        X = torch.stack(input_tensors, dim=0).float() 

        # Target (Shape: N)
        Y = torch.from_numpy(self._load_morph_data(sample["target"])).long()
        if self.map_to_unknown:
            mask = torch.isin(Y, self.labels_to_replace)
            Y[mask] = 2

        # Subject-level Z-Score normalization (excluding channel 0)
        if X.shape[0] > 1:
            features = X[1:, :]
            mean = features.mean(dim=1, keepdim=True)
            std = features.std(dim=1, keepdim=True)
            X[1:, :] = (features - mean) / (std + 1e-8)

        # Augmentations (transform handles the ch0 vs others logic)
        if self.transform:
            _, _, X, Y = self.transform(self.coords, self.faces, X, Y, cfg=self.cfg)

        return self.coords, self.faces, X, Y