import torch
import torch.nn as nn
from surfify.models.base import SphericalBase
from surfify.nn import IcoDiNeConv, IcoPool, IcoUpSample
from surfify.utils import number_of_ico_vertices
#import logging
from surfify.models import SphericalUNet
import time 
from surfify.utils import icosahedron
import numpy as np 





import segmentation_models_pytorch as smp

# Configuration du logger
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

class MySphericalUNet(SphericalBase):
    def __init__(self, in_order, in_channels, out_channels, base_filters=16, depth=3):
        
        super().__init__(
            input_order=in_order,
            n_layers=depth,
            conv_mode="DiNe",
            dine_size=1,
            standard_ico=True,
            cachedir="../spherical_labelling/models/surfify_cache" 
        )
        #logging.debug("Initialisation de MySphericalUNet")
        print ("ok")
        self.depth = depth
        self.in_vertices = number_of_ico_vertices(in_order)
        self.filts = [in_channels] + [base_filters * 2**i for i in range(depth)]

        # Encoder

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            conv = IcoDiNeConv(self.filts[i], self.filts[i+1], self.ico[in_order - i].neighbor_indices)
            self.down_convs.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(self.filts[i+1]),
                nn.ReLU()
            ))
            if i < depth - 1:
                pool = IcoPool(
                    down_neigh_indices=self.ico[in_order - i].neighbor_indices,
                    down_indices=self.ico[in_order - i].down_indices,
                    pooling_type="mean"
                )
                self.pools.append(pool)

        # Decoder
        print ("start decode")
        self.upsamples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            up = IcoUpSample(self.filts[i+2], self.filts[i+1], self.ico[in_order - i - 1].up_indices)
            conv = IcoDiNeConv(self.filts[i+2], self.filts[i+1], self.ico[in_order - i - 1].neighbor_indices)
            self.upsamples.append(up)
            self.up_convs.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(self.filts[i+1]),
                nn.ReLU()
            ))

        # Final 1x1 convolution
        self.final = nn.Conv1d(self.filts[1], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_feats = []
        # Encoder path
        for i in range(self.depth):
            x = self.down_convs[i](x)
            encoder_feats.append(x)
            if i < self.depth - 1:
                x, _ = self.pools[i](x)

        # Decoder path
        for i in range(self.depth - 1):
            x = self.upsamples[i](x)
            skip = encoder_feats[self.depth - 2 - i]
            x = torch.cat([x, skip], dim=1)
            x = self.up_convs[i](x)

        x = self.final(x)
        return x



class Spherical2DCNN(nn.Module):

    def __init__(self, in_channels, out_channels, img_size=330, ico_order=6, encoder_name="resnet18"):
        super().__init__()
        
        # --- 1. CONFIGURATION CUBEMAP ---
        # Ajustement de la taille pour la divisibilité par 32 (ResNet stride)
        raw_face_size = img_size // 3
        self.face_size = int(round(raw_face_size / 32) * 32) 
        
        self.H_total = self.face_size * 2
        self.W_total = self.face_size * 3
        self.img_size = (self.H_total, self.W_total)
        
        print(f"📦 Cubemap Configuration: 6 faces de {self.face_size}px")
        print(f"🖼️ Full Image Size for Unet: {self.img_size}")

        # --- 2. PRE-CALCUL DU MAPPING ---

        sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
        self.N = sphere_xyz.shape[0]
        
        # A. Coordonnées 3D des pixels de la Cubemap
        pixel_xyz = self._generate_cubemap_coords(self.face_size)
        
        # B. KDTree pour le Forward (Sphere -> Image)
        print("⏳ Building KDTree for Spherical Mapping...")
        tree = cKDTree(sphere_xyz)
        _, nearest_vertex_indices = tree.query(pixel_xyz, k=1)
        self.register_buffer('pixel_to_vertex_idx', torch.from_numpy(nearest_vertex_indices).long())
        
        # C. Calcul des UV pour le Backward (Image -> Sphere)
        vertex_uv_np = self._compute_vertex_uv(sphere_xyz)
        self.register_buffer('vertex_uv', torch.from_numpy(vertex_uv_np).float())
        print("✅ Geometry Mapping Initialized.")

        # --- 3. ARCHITECTURE U-NET ---
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels
        )

    def _generate_cubemap_coords(self, S):
        """Génère les coordonnées XYZ normalisées pour chaque pixel de la grille 2x3."""
        rng = np.linspace(-1, 1, S)
        grid_u, grid_v = np.meshgrid(rng, rng)
        
        xyz_list = []
        xyz_list.append(np.stack([np.ones_like(grid_u), -grid_v, -grid_u], axis=-1)) # Right
        xyz_list.append(np.stack([-np.ones_like(grid_u), -grid_v, grid_u], axis=-1)) # Left
        xyz_list.append(np.stack([grid_u, np.ones_like(grid_u), grid_v], axis=-1))   # Top
        xyz_list.append(np.stack([grid_u, -np.ones_like(grid_u), -grid_v], axis=-1)) # Bottom
        xyz_list.append(np.stack([grid_u, -grid_v, np.ones_like(grid_u)], axis=-1))  # Front
        xyz_list.append(np.stack([-grid_u, -grid_v, -np.ones_like(grid_u)], axis=-1))# Back
        
        row1 = np.concatenate(xyz_list[0:3], axis=1) 
        row2 = np.concatenate(xyz_list[3:6], axis=1)
        full_grid_xyz = np.concatenate([row1, row2], axis=0)
        
        norm = np.linalg.norm(full_grid_xyz, axis=-1, keepdims=True)
        return (full_grid_xyz / norm).reshape(-1, 3)

    def _compute_vertex_uv(self, sphere_xyz):
        """Mapping inverse : coordonnées (u, v) de chaque sommet dans l'image."""
        x, y, z = sphere_xyz.T
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        is_y = (abs_y > abs_x) & (abs_y >= abs_z)
        is_z = (abs_z > abs_x) & (abs_z > abs_y)
        
        face_idx = np.zeros(self.N, dtype=int)
        u_loc, v_loc = np.zeros(self.N), np.zeros(self.N)
        
        m = is_x & (x > 0); face_idx[m]=0; u_loc[m]=-z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_x & (x < 0); face_idx[m]=1; u_loc[m]=z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_y & (y > 0); face_idx[m]=2; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=z[m]/abs_y[m]
        m = is_y & (y < 0); face_idx[m]=3; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=-z[m]/abs_y[m]
        m = is_z & (z > 0); face_idx[m]=4; u_loc[m]=x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]
        m = is_z & (z < 0); face_idx[m]=5; u_loc[m]=-x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]

        col, row = face_idx % 3, face_idx // 3
        u_px = (u_loc + 1) * 0.5 * (self.face_size - 1)
        v_px = (v_loc + 1) * 0.5 * (self.face_size - 1)
        
        return np.stack([u_px + (col * self.face_size), v_px + (row * self.face_size)], axis=1)

    def forward(self, x):
        """
        x: [Batch, Channels, N_vertices]
        """
        B, C, N = x.shape
        
        # 1. Projection vers Image Cubemap
        idx = self.pixel_to_vertex_idx.view(1, 1, -1).expand(B, C, -1)
        x_img = torch.gather(x, 2, idx).view(B, C, self.H_total, self.W_total)
        
        # 2. Forward U-Net
        logits_2d = self.unet(x_img)
        
        # 3. Back-projection vers la Sphère
        u = self.vertex_uv[:, 0].long()
        v = self.vertex_uv[:, 1].long()
        
        logits_sphere = logits_2d[:, :, v, u] 
        
        return logits_sphere # [B, out_channels, N]



class SphericalUNetFromPretrained(nn.Module):
    def __init__(self, backbone, num_classes, freeze_encoder=True, use_3_channels=False, dropout=0.25, use_skip_adapter=False):
        super().__init__()
        
        self.backbone = backbone
        self.depth = backbone.num_blocks
        self.use_3_channels = use_3_channels
        self.use_skip_adapter=use_skip_adapter

        # --- ADAPTATEUR 3 vers 2 CANAUX ---
        if self.use_3_channels:
            self.input_adapter = nn.Conv1d(3, 2, kernel_size=1, bias=False)
            
            with torch.no_grad():
                self.input_adapter.weight[0, 0, 0] = 0.1
                self.input_adapter.weight[0, 1, 0] = 0.9
                self.input_adapter.weight[0, 2, 0] = 0.0

                self.input_adapter.weight[1, 0, 0] = 0.1
                self.input_adapter.weight[1, 1, 0] = 0.0
                self.input_adapter.weight[1, 2, 0] = 0.9
                
            print("🔌 Adaptateur 3->2 canaux activé et initialisé (0.9/0.1).")
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Encoder (Backbone) figé.")

        # 2. Construction du Décodeur (Symétrique)
        self.upsamples = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        
        self.skip_adapters = nn.ModuleList()
        current_ico = backbone.input_order 
        
        in_ch = backbone.start_channels * (2 ** (self.depth - 1))
        
        for i in range(self.depth - 1):
            out_ch = in_ch // 2
            
            # --- CREATION DE L'ADAPTATEUR POUR CE NIVEAU ---
            if self.use_skip_adapter:
                adapter = nn.Sequential(
                    nn.Conv1d(out_ch, out_ch, kernel_size=1),
                    nn.GroupNorm(8, out_ch),
                    nn.LeakyReLU(0.1)
                )
                self.skip_adapters.append(adapter)
            else:
                self.skip_adapters.append(nn.Identity())

            decode_order_idx = (backbone.input_order - self.depth + 1) + i
            
            up = IcoUpSample(in_ch, out_ch, backbone.ico[decode_order_idx].up_indices)
            
            conv = nn.Sequential(
                IcoDiNeConv(in_ch, out_ch, backbone.ico[decode_order_idx + 1].neighbor_indices),
                nn.GroupNorm(16, out_ch),
       
                nn.LeakyReLU(0.15),
                nn.Dropout(dropout)
            )
            
            self.upsamples.append(up)
            self.dec_convs.append(conv)
            
            in_ch = out_ch

        # 3. Final classifier (1x1 Conv)
        self.final = nn.Conv1d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        if self.use_3_channels:
            x = self.input_adapter(x)
        skips = []
        
        # --- ENCODER ---
        layer_idx = 0
        layers = self.backbone.encoder_layers
        
        for i in range(self.depth):
            # Conv 1
            x = layers[layer_idx](x); layer_idx += 1
            layers[layer_idx].eval()
            x = layers[layer_idx](x); layer_idx += 1
            x = layers[layer_idx](x); layer_idx += 1
            
            # Conv 2
            x = layers[layer_idx](x); layer_idx += 1
            layers[layer_idx].eval()
            x = layers[layer_idx](x); layer_idx += 1
            x = layers[layer_idx](x); layer_idx += 1
            
            if i < self.depth - 1:
                skips.append(x)
                x, _ = layers[layer_idx](x)
                layer_idx += 1
        
        # --- DECODER ---
        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            
            skip_val = skips.pop()

            skip_val = self.skip_adapters[i](skip_val)
            
            x = torch.cat([x, skip_val], dim=1)
            
            x = self.dec_convs[i](x)
            
        # --- FINAL ---
        x = self.final(x)
        return x

import torch.nn.functional as F
import numpy as np
from surfify.utils import icosahedron

# class SphericalDinoV2(nn.Module):
#     def __init__(self, in_channels, out_channels, img_size=518, ico_order=6, dino_size="dinov2_vits14"):
#         super().__init__()
        
#         if img_size % 14 != 0:
#             img_size = int(round(img_size / 14) * 14)
#             print(f"⚠️ Image size ajustée à {img_size}x{img_size} pour DINO.")

#         self.img_size = (img_size, img_size)
        
#         sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
#         self.N = sphere_xyz.shape[0] 
        
#         x, y, z = sphere_xyz.T
#         theta = np.arccos(np.clip(z, -1, 1)) 
#         phi = np.arctan2(y, x)               
        
#         H, W = self.img_size
#         u = ((phi + np.pi) / (2 * np.pi)) * (W - 1)
#         v = (theta / np.pi) * (H - 1)
#         uv = np.stack([u, v], axis=1)
        
#         self.register_buffer('sphere_uv', torch.tensor(uv, dtype=torch.long))

#         print(f"🦖 Chargement de {dino_size} pour sphère de {self.N} sommets...")
#         self.dino = torch.hub.load('facebookresearch/dinov2', dino_size)
        
#         for param in self.dino.parameters():
#             param.requires_grad = False
#         self.dino.eval() 

#         self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        
#         self.embed_dim = self.dino.embed_dim 
#         self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

#     def project_sphere_to_image(self, x):
#         B, C, N = x.shape
#         H, W = self.img_size
        
#         if N != self.N:
#             raise ValueError(f"Erreur dimension: Input a {N} sommets, mais le modèle attend {self.N} (Ordre défini à l'init).")

#         uv = self.sphere_uv.clamp(min=0)
#         u = uv[:, 0].clamp(0, W - 1)
#         v = uv[:, 1].clamp(0, H - 1)
#         idx_flat = (v * W + u) 
        
#         img = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)
        
#         index = idx_flat.unsqueeze(0).expand(C, -1) 
        
#         for b in range(B):
#             img[b].scatter_(dim=1, index=index, src=x[b])
            
#         return img.view(B, C, H, W)

#     def backproject_logits_to_sphere(self, logits_2d):
#         B, C, H, W = logits_2d.shape
        
#         u = self.sphere_uv[:, 0].clamp(0, W - 1)
#         v = self.sphere_uv[:, 1].clamp(0, H - 1)
        
#         logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
        
#         for b in range(B):
#             logits_sphere[b] = logits_2d[b, :, v, u]

#         return logits_sphere

#  
#     def forward(self, x):
        
#         x_img = self.project_sphere_to_image(x) 
        
#         x_rgb = self.input_adapter(x_img)
        
#         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
#         x_norm = (x_rgb - mean) / std

#         with torch.no_grad():
#             out = self.dino.forward_features(x_norm)
#             patch_tokens = out['x_norm_patchtokens'] 

#         B, N_patches, Dim = patch_tokens.shape
#         H_grid = W_grid = int(N_patches**0.5) 
#         features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
        
#         logits_low_res = self.head(features_2d)
        
#         logits_high_res = F.interpolate(
#             logits_low_res, 
#             size=self.img_size, 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         logits_sphere = self.backproject_logits_to_sphere(logits_high_res)
        
#         return logits_sphere




# class SphericalDinoV2(nn.Module):
#     def __init__(self, in_channels, out_channels, img_size=518, ico_order=6, dino_size="dinov2_vits14",unfreeze_blocks=0):
#         super().__init__()
        
#         raw_face_size = img_size // 3
#         self.face_size = int(round(raw_face_size / 14) * 14)
        
#         self.H_total = self.face_size * 2  
#         self.W_total = self.face_size * 3  
#         self.img_size = (self.H_total, self.W_total)
        
#         print(f"📦 Cubemap Projection: 6 faces de {self.face_size}px")
#         print(f"🖼️ Image envoyée à DINO: {self.img_size} (Grille 2x3)")

#         sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
#         self.N = sphere_xyz.shape[0]
        
#         x, y, z = sphere_xyz.T
#         abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
#         is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
#         is_y = (abs_y > abs_x) & (abs_y >= abs_z)
#         is_z = (abs_z > abs_x) & (abs_z > abs_y)
        
#         face_idx = np.zeros(self.N, dtype=int)
#         u_local = np.zeros(self.N)
#         v_local = np.zeros(self.N)
        
#         mask = is_x & (x > 0); face_idx[mask] = 0; u_local[mask] = -z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
#         mask = is_x & (x < 0); face_idx[mask] = 1; u_local[mask] = z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
#         mask = is_y & (y > 0); face_idx[mask] = 2; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = z[mask]/abs_y[mask]
#         mask = is_y & (y < 0); face_idx[mask] = 3; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = -z[mask]/abs_y[mask]
#         mask = is_z & (z > 0); face_idx[mask] = 4; u_local[mask] = x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]
#         mask = is_z & (z < 0); face_idx[mask] = 5; u_local[mask] = -x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]

#         u_px = (u_local + 1) * 0.5 * (self.face_size - 1)
#         v_px = (v_local + 1) * 0.5 * (self.face_size - 1)
        
#         col = face_idx % 3
#         row = face_idx // 3
        
#         u_final = u_px + (col * self.face_size)
#         v_final = v_px + (row * self.face_size)
        
#         uv = np.stack([u_final, v_final], axis=1)
#         self.register_buffer('sphere_uv', torch.tensor(uv, dtype=torch.long))

#         print(f"🦖 Chargement de {dino_size}...")
#         self.dino = torch.hub.load('facebookresearch/dinov2', dino_size)
        
#         for param in self.dino.parameters():
#             param.requires_grad = False
#         self.dino.eval() 

#         self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        
#         self.embed_dim = self.dino.embed_dim 
#         self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

#     def project_sphere_to_image(self, x):
#         B, C, N = x.shape
#         H, W = self.img_size
        
#         uv = self.sphere_uv.clamp(min=0)
#         u = uv[:, 0].clamp(0, W - 1)
#         v = uv[:, 1].clamp(0, H - 1)
#         idx_flat = (v * W + u) 
        
#         img = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)
#         index = idx_flat.unsqueeze(0).expand(C, -1) 
        
#         for b in range(B):
#             img[b].scatter_(dim=1, index=index, src=x[b])
            
#         return img.view(B, C, H, W)

#     def backproject_logits_to_sphere(self, logits_2d):
#         B, C, H, W = logits_2d.shape
        
#         u = self.sphere_uv[:, 0].clamp(0, W - 1)
#         v = self.sphere_uv[:, 1].clamp(0, H - 1)
        
#         logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
        
#         for b in range(B):
#             logits_sphere[b] = logits_2d[b, :, v, u]

#         return logits_sphere

#     def forward(self, x):
#         x_img = self.project_sphere_to_image(x) 

        
        
#         x_rgb = self.input_adapter(x_img)
        
#         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
#         x_norm = (x_rgb - mean) / std

#         with torch.no_grad():
#             out = self.dino.forward_features(x_norm)
#             patch_tokens = out['x_norm_patchtokens'] 

#         B, N_patches, Dim = patch_tokens.shape
        
#         H_grid = self.H_total // 14
#         W_grid = self.W_total // 14
        
#         features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
        
#         logits_low_res = self.head(features_2d) 
        
#         logits_high_res = F.interpolate(
#             logits_low_res, 
#             size=self.img_size, 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         logits_sphere = self.backproject_logits_to_sphere(logits_high_res)
        
#         return logits_sphere


from scipy.spatial import cKDTree

class SphericalDinoV2(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=518, ico_order=6, dino_size="dinov2_vits14", unfreeze_blocks=0):
        super().__init__()
        
        # --- 1. CONFIGURATION DE L'IMAGE CUBEMAP ---
        raw_face_size = img_size // 3
        self.face_size = int(round(raw_face_size / 14) * 14) 
        
        self.H_total = self.face_size * 2
        self.W_total = self.face_size * 3
        self.img_size = (self.H_total, self.W_total)
        
        print(f"📦 Cubemap (Scipy Filled): 6 faces de {self.face_size}px")
        print(f"🖼️ Image DINO: {self.img_size}")

        # --- 2. PRE-CALCUL DU MAPPING ---
        sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
        self.N = sphere_xyz.shape[0]
        
        # A. Coordonnées XYZ des pixels cibles
        pixel_xyz, _ = self._generate_cubemap_coords(self.face_size)
        
        # B. Calcul des UV des sommets
        vertex_uv_np = self._compute_vertex_uv(sphere_xyz)
        self.register_buffer('vertex_uv', torch.from_numpy(vertex_uv_np).float())

        # C. Mapping KDTree
        print("⏳ Construction du KDTree (Scipy)...")
        tree = cKDTree(sphere_xyz)
        _, nearest_vertex_indices = tree.query(pixel_xyz, k=1)
        
        self.register_buffer('pixel_to_vertex_idx', torch.from_numpy(nearest_vertex_indices).long())
        
        print("✅ Mapping calculé.")

        # --- 3. DINOv2 ---
        print(f"🦖 Chargement de {dino_size}...")
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_size)

        for param in self.dino.parameters():
            param.requires_grad = False
        
        # Logique de dégel
        if unfreeze_blocks != 0:
            total_blocks = len(self.dino.blocks)
            
            if unfreeze_blocks == -1:
                print("🔥 Attention: DINOv2 est totalement dégelé (Full Fine-Tuning)!")
                start_idx = 0
            else:
                print(f"❄️/🔥 Dégel partiel : On entraîne les {unfreeze_blocks} derniers blocs sur {total_blocks}.")
                start_idx = total_blocks - unfreeze_blocks

            for i, block in enumerate(self.dino.blocks):
                print (block)
                if i >= start_idx:
                    for param in block.parameters():
                        param.requires_grad = True
            
            if hasattr(self.dino, 'norm'):
                for param in self.dino.norm.parameters():
                    param.requires_grad = True
            
        else:
            print("❄️ DINOv2 est totalement gelé (Feature Extractor uniquement).")
            self.dino.eval()

        # --- 4. HEADS ---
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.embed_dim = self.dino.embed_dim 
        self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

    def _generate_cubemap_coords(self, S):
        """Génère les coordonnées 3D pour chaque pixel de la grille 2x3."""
        rng = np.linspace(-1, 1, S)
        grid_u, grid_v = np.meshgrid(rng, rng) 
        
        xyz_list = []
        xyz_list.append(np.stack([np.ones_like(grid_u), -grid_v, -grid_u], axis=-1)) # Right
        xyz_list.append(np.stack([-np.ones_like(grid_u), -grid_v, grid_u], axis=-1)) # Left
        xyz_list.append(np.stack([grid_u, np.ones_like(grid_u), grid_v], axis=-1))   # Top
        xyz_list.append(np.stack([grid_u, -np.ones_like(grid_u), -grid_v], axis=-1)) # Bottom
        xyz_list.append(np.stack([grid_u, -grid_v, np.ones_like(grid_u)], axis=-1))  # Front
        xyz_list.append(np.stack([-grid_u, -grid_v, -np.ones_like(grid_u)], axis=-1))# Back
        
        row1 = np.concatenate(xyz_list[0:3], axis=1) 
        row2 = np.concatenate(xyz_list[3:6], axis=1)
        full_grid_xyz = np.concatenate([row1, row2], axis=0)
        
        norm = np.linalg.norm(full_grid_xyz, axis=-1, keepdims=True)
        full_grid_xyz = full_grid_xyz / norm
        
        return full_grid_xyz.reshape(-1, 3), None

    def _compute_vertex_uv(self, sphere_xyz):
        """Calcule (u, v) exacts pour chaque sommet."""
        x, y, z = sphere_xyz.T
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        is_y = (abs_y > abs_x) & (abs_y >= abs_z)
        is_z = (abs_z > abs_x) & (abs_z > abs_y)
        
        face_idx = np.zeros(self.N, dtype=int)
        u_loc, v_loc = np.zeros(self.N), np.zeros(self.N)
        
        m = is_x & (x > 0); face_idx[m]=0; u_loc[m]=-z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_x & (x < 0); face_idx[m]=1; u_loc[m]=z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_y & (y > 0); face_idx[m]=2; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=z[m]/abs_y[m]
        m = is_y & (y < 0); face_idx[m]=3; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=-z[m]/abs_y[m]
        m = is_z & (z > 0); face_idx[m]=4; u_loc[m]=x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]
        m = is_z & (z < 0); face_idx[m]=5; u_loc[m]=-x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]

        col = face_idx % 3
        row = face_idx // 3
        
        u_px = (u_loc + 1) * 0.5 * (self.face_size - 1)
        v_px = (v_loc + 1) * 0.5 * (self.face_size - 1)
        
        u_final = u_px + (col * self.face_size)
        v_final = v_px + (row * self.face_size)
        
        return np.stack([u_final, v_final], axis=1)

    def project_sphere_to_image(self, x):
        """Projection Sphère vers Image via Indexing."""
        B, C, N = x.shape
        idx = self.pixel_to_vertex_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        
        flat_img = torch.gather(x, 2, idx) 
        
        return flat_img.view(B, C, self.H_total, self.W_total)



    def backproject_logits_to_sphere(self, logits_2d):
        """Image vers Sphère via UV Sampling."""
        B, C, H, W = logits_2d.shape
        
        u = self.vertex_uv[:, 0].clamp(0, W - 1).long()
        v = self.vertex_uv[:, 1].clamp(0, H - 1).long()
        
        logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
        for b in range(B):
            logits_sphere[b] = logits_2d[b, :, v, u]
            
        return logits_sphere


    def forward(self, x):
        # 1. Projection
        x_img = self.project_sphere_to_image(x)

        # 2. Adaptation RGB & Norm
        x_rgb = self.input_adapter(x_img)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_rgb - mean) / std

        # 3. DINOv2 Features
        with torch.no_grad():
            out = self.dino.forward_features(x_norm)
            patch_tokens = out['x_norm_patchtokens']

        # 4. Reshape & Head
        B, N_patches, Dim = patch_tokens.shape
        H_grid = self.H_total // 14
        W_grid = self.W_total // 14
        
        features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
        
        logits_low_res = self.head(features_2d)
        
        # 5. Upsampling
        logits_high_res = F.interpolate(
            logits_low_res, 
            size=self.img_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        # 6. Backprojection
        logits_sphere = self.backproject_logits_to_sphere(logits_high_res)
        
        return logits_sphere


from transformers import AutoModel, AutoConfig 

# class SphericalDinoV3(nn.Module):
#     def __init__(self, in_channels, out_channels, img_size=528, ico_order=6, 
#                  dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m", 
#                  patch_size=16):
#         super().__init__()
        
#         self.patch_size = patch_size
#         raw_face_size = img_size // 3
#         self.face_size = int(round(raw_face_size / self.patch_size) * self.patch_size)
        
#         self.H_total = self.face_size * 2
#         self.W_total = self.face_size * 3
#         self.img_size = (self.H_total, self.W_total)
        
#         print(f"📦 DINOv3 Cubemap (HF): 6 faces de {self.face_size}px")
#         print(f"🖼️ Image envoyée: {self.img_size}")

#         sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
#         self.N = sphere_xyz.shape[0]
        
#         x, y, z = sphere_xyz.T
#         abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
#         is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
#         is_y = (abs_y > abs_x) & (abs_y >= abs_z)
#         is_z = (abs_z > abs_x) & (abs_z > abs_y)
#         face_idx = np.zeros(self.N, dtype=int); u_local = np.zeros(self.N); v_local = np.zeros(self.N)
        
#         mask = is_x & (x > 0); face_idx[mask] = 0; u_local[mask] = -z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
#         mask = is_x & (x < 0); face_idx[mask] = 1; u_local[mask] = z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
#         mask = is_y & (y > 0); face_idx[mask] = 2; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = z[mask]/abs_y[mask]
#         mask = is_y & (y < 0); face_idx[mask] = 3; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = -z[mask]/abs_y[mask]
#         mask = is_z & (z > 0); face_idx[mask] = 4; u_local[mask] = x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]
#         mask = is_z & (z < 0); face_idx[mask] = 5; u_local[mask] = -x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]

#         u_px = (u_local + 1) * 0.5 * (self.face_size - 1)
#         v_px = (v_local + 1) * 0.5 * (self.face_size - 1)
#         col = face_idx % 3; row = face_idx // 3
#         u_final = u_px + (col * self.face_size); v_final = v_px + (row * self.face_size)
#         uv = np.stack([u_final, v_final], axis=1)
#         self.register_buffer('sphere_uv', torch.tensor(uv, dtype=torch.long))


#         mon_token = "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" 

#         print(f"🦖 Chargement de {dino_model_name} via Hugging Face...")

#         self.config = AutoConfig.from_pretrained(dino_model_name, trust_remote_code=True, token=mon_token)
#         self.num_registers = getattr(self.config, "num_register_tokens", 0)
#         self.embed_dim = self.config.hidden_size

#         self.dino = AutoModel.from_pretrained(dino_model_name, trust_remote_code=True, token=mon_token)
        
#         for param in self.dino.parameters():
#             param.requires_grad = False
#         self.dino.eval() 
        
#         print(f"   🔹 Embed Dim: {self.embed_dim}")
#         print(f"   🔹 Registers: {self.num_registers}")

#         self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
#         self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

#     def project_sphere_to_image(self, x):
#         B, C, N = x.shape
#         H, W = self.img_size
#         uv = self.sphere_uv.clamp(min=0)
#         u = uv[:, 0].clamp(0, W - 1); v = uv[:, 1].clamp(0, H - 1)
#         idx_flat = (v * W + u)
#         img = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)
#         index = idx_flat.unsqueeze(0).expand(C, -1)
#         for b in range(B):
#             img[b].scatter_(dim=1, index=index, src=x[b])
#         return img.view(B, C, H, W)

#     def backproject_logits_to_sphere(self, logits_2d):
#         B, C, H, W = logits_2d.shape
#         u = self.sphere_uv[:, 0].clamp(0, W - 1); v = self.sphere_uv[:, 1].clamp(0, H - 1)
#         logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
#         for b in range(B):
#             logits_sphere[b] = logits_2d[b, :, v, u]
#         return logits_sphere

#     def forward(self, x):
#         x_img = self.project_sphere_to_image(x)
        
#         x_rgb = self.input_adapter(x_img)
#         mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
#         x_norm = (x_rgb - mean) / std

#         with torch.no_grad():
#             outputs = self.dino(pixel_values=x_norm)
            
#             last_hidden = outputs.last_hidden_state 
            
#             start_idx = 1 + self.num_registers
#             patch_tokens = last_hidden[:, start_idx:, :] 

#         B, N_patches, Dim = patch_tokens.shape
#         H_grid = self.H_total // self.patch_size
#         W_grid = self.W_total // self.patch_size
        
#         features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
        
#         logits_low_res = self.head(features_2d)
        
#         logits_high_res = F.interpolate(
#             logits_low_res, 
#             size=self.img_size, 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         return self.backproject_logits_to_sphere(logits_high_res)

class LightweightFPNDecoder(nn.Module):
    """
    Décodeur FPN léger : fusion de 4 couches ViT (1/16) et upsampling progressif.
    """
    def __init__(self, embed_dim, out_channels, feature_dim=256):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Conv2d(embed_dim, feature_dim, kernel_size=1) for _ in range(4)
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = self._make_up_block(feature_dim, feature_dim // 2)
        self.up2 = self._make_up_block(feature_dim // 2, feature_dim // 4)
        self.up3 = self._make_up_block(feature_dim // 4, feature_dim // 8)
        self.up4 = self._make_up_block(feature_dim // 8, feature_dim // 8)
        
        self.head = nn.Conv2d(feature_dim // 8, out_channels, kernel_size=1)

    def _make_up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list):
        projs = [proj(feat) for proj, feat in zip(self.projections, features_list)]
        
        fused = torch.cat(projs, dim=1) 
        x = self.fusion(fused)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        return self.head(x)


class SphericalDinoV3(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=528, ico_order=6, 
                 dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                 patch_size=16):
        super().__init__()
        
        # --- 1. CONFIGURATION CUBEMAP ---
        self.patch_size = patch_size
        raw_face_size = img_size // 3
        self.face_size = int(round(raw_face_size / self.patch_size) * self.patch_size)
        
        self.H_total = self.face_size * 2
        self.W_total = self.face_size * 3
        self.img_size = (self.H_total, self.W_total)
        
        print(f"📦 DINOv3 Cubemap (HF): 6 faces de {self.face_size}px")
        print(f"🖼️ Image envoyée: {self.img_size}")

        sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
        self.N = sphere_xyz.shape[0]
        
        x, y, z = sphere_xyz.T
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        is_y = (abs_y > abs_x) & (abs_y >= abs_z)
        is_z = (abs_z > abs_x) & (abs_z > abs_y)
        face_idx = np.zeros(self.N, dtype=int); u_local = np.zeros(self.N); v_local = np.zeros(self.N)
        
        mask = is_x & (x > 0); face_idx[mask] = 0; u_local[mask] = -z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
        mask = is_x & (x < 0); face_idx[mask] = 1; u_local[mask] = z[mask]/abs_x[mask]; v_local[mask] = -y[mask]/abs_x[mask]
        mask = is_y & (y > 0); face_idx[mask] = 2; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = z[mask]/abs_y[mask]
        mask = is_y & (y < 0); face_idx[mask] = 3; u_local[mask] = x[mask]/abs_y[mask]; v_local[mask] = -z[mask]/abs_y[mask]
        mask = is_z & (z > 0); face_idx[mask] = 4; u_local[mask] = x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]
        mask = is_z & (z < 0); face_idx[mask] = 5; u_local[mask] = -x[mask]/abs_z[mask]; v_local[mask] = -y[mask]/abs_z[mask]

        u_px = (u_local + 1) * 0.5 * (self.face_size - 1)
        v_px = (v_local + 1) * 0.5 * (self.face_size - 1)
        col = face_idx % 3; row = face_idx // 3
        u_final = u_px + (col * self.face_size); v_final = v_px + (row * self.face_size)
        uv = np.stack([u_final, v_final], axis=1)
        self.register_buffer('sphere_uv', torch.tensor(uv, dtype=torch.long))

        # --- 2. BACKBONE DINOv3 via TRANSFORMERS ---
        mon_token = "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" 
        print(f"🦖 Chargement de {dino_model_name} via Hugging Face...")

        self.config = AutoConfig.from_pretrained(
            dino_model_name, trust_remote_code=True, token=mon_token, 
            output_hidden_states=True 
        )
        self.num_registers = getattr(self.config, "num_register_tokens", 0)
        self.embed_dim = self.config.hidden_size
        
        num_layers = self.config.num_hidden_layers
        self.out_indices = [
            (num_layers // 4) - 1, 
            (num_layers // 2) - 1, 
            (num_layers * 3 // 4) - 1, 
            num_layers - 1
        ]
        print(f"   🔹 Extraction multi-échelles aux couches : {self.out_indices}")

        self.dino = AutoModel.from_pretrained(
            dino_model_name, trust_remote_code=True, token=mon_token,
            output_hidden_states=True 
        )
        
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval() 
        
        print(f"   🔹 Embed Dim: {self.embed_dim}")
        print(f"   🔹 Registers: {self.num_registers}")

        # --- 3. ADAPTERS ET DECODEUR ---
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        
        self.decoder = LightweightFPNDecoder(embed_dim=self.embed_dim, out_channels=out_channels)

    def project_sphere_to_image(self, x):
        B, C, N = x.shape
        H, W = self.img_size
        uv = self.sphere_uv.clamp(min=0)
        u = uv[:, 0].clamp(0, W - 1); v = uv[:, 1].clamp(0, H - 1)
        idx_flat = (v * W + u)
        img = torch.zeros(B, C, H * W, device=x.device, dtype=x.dtype)
        index = idx_flat.unsqueeze(0).expand(C, -1)
        for b in range(B):
            img[b].scatter_(dim=1, index=index, src=x[b])
        return img.view(B, C, H, W)

    def backproject_logits_to_sphere(self, logits_2d):
        B, C, H, W = logits_2d.shape
        u = self.sphere_uv[:, 0].clamp(0, W - 1); v = self.sphere_uv[:, 1].clamp(0, H - 1)
        logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
        for b in range(B):
            logits_sphere[b] = logits_2d[b, :, v, u]
        return logits_sphere

    def forward(self, x):
        x_img = self.project_sphere_to_image(x)
        
        x_rgb = self.input_adapter(x_img)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_rgb - mean) / std

        with torch.no_grad():
            outputs = self.dino(pixel_values=x_norm)
            all_hidden_states = outputs.hidden_states 

        H_grid = self.H_total // self.patch_size
        W_grid = self.W_total // self.patch_size
        
        features_2d_list = []
        for idx in self.out_indices:
            layer_out = all_hidden_states[idx + 1] 
            
            start_idx = 1 + self.num_registers
            patch_tokens = layer_out[:, start_idx:, :] 
            
            B, _, Dim = patch_tokens.shape
            features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
            features_2d_list.append(features_2d)
        
        logits_high_res = self.decoder(features_2d_list)
        
        return self.backproject_logits_to_sphere(logits_high_res)


class SphericalDinoV3Linear(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=528, ico_order=6, 
                 dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                 patch_size=16, unfreeze_blocks=0):
        super().__init__()
        
        # --- 1. CONFIGURATION CUBEMAP ---
        self.patch_size = patch_size
        raw_face_size = img_size // 3
        self.face_size = int(round(raw_face_size / self.patch_size) * self.patch_size)
        
        self.H_total = self.face_size * 2
        self.W_total = self.face_size * 3
        self.img_size = (self.H_total, self.W_total)
        
        print(f"📦 DINOv3 Cubemap (HF): 6 faces de {self.face_size}px")
        print(f"🖼️ Image envoyée: {self.img_size}")

        sphere_xyz, _ = icosahedron(order=ico_order, standard_ico=True)
        self.N = sphere_xyz.shape[0]

        pixel_xyz, _ = self._generate_cubemap_coords(self.face_size)
        
        vertex_uv_np = self._compute_vertex_uv(sphere_xyz)
        self.register_buffer('vertex_uv', torch.from_numpy(vertex_uv_np).float())

        print("⏳ Construction du KDTree (Scipy)...")
        tree = cKDTree(sphere_xyz)
        _, nearest_vertex_indices = tree.query(pixel_xyz, k=1)
        self.register_buffer('pixel_to_vertex_idx', torch.from_numpy(nearest_vertex_indices).long())
        print("✅ Mapping calculé.")

        # --- 2. BACKBONE DINOv3 via TRANSFORMERS ---
        mon_token = "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" 
        print(f"🦖 Chargement de {dino_model_name} via Hugging Face...")

        self.config = AutoConfig.from_pretrained(
            dino_model_name, trust_remote_code=True, token=mon_token, 
            output_hidden_states=True 
        )
        self.num_registers = getattr(self.config, "num_register_tokens", 0)
        self.embed_dim = self.config.hidden_size

        self.dino = AutoModel.from_pretrained(
            dino_model_name, config=self.config, trust_remote_code=True, token=mon_token
        )

        for param in self.dino.parameters():
            param.requires_grad = False
        
        self.unfreeze_blocks = unfreeze_blocks
        if unfreeze_blocks != 0:
            if hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
                blocks = self.dino.encoder.layer
            elif hasattr(self.dino, 'layers'):
                blocks = self.dino.layers
            elif hasattr(self.dino, 'blocks'):
                blocks = self.dino.blocks
            elif hasattr(self.dino, 'vision_model') and hasattr(self.dino.vision_model.encoder, 'layer'):
                blocks = self.dino.vision_model.encoder.layer
            else:
                module_lists = [m for m in self.dino.modules() if isinstance(m, nn.ModuleList)]
                if not module_lists:
                    raise AttributeError("Impossible de trouver les blocs. Ajoute `print(self.dino)` juste avant cette ligne pour inspecter l'architecture exacte.")
                blocks = max(module_lists, key=lambda ml: len(ml))

            total_blocks = len(blocks)
            
            if unfreeze_blocks == -1:
                print("🔥 Attention: DINOv3 est totalement dégelé (Full Fine-Tuning)!")
                start_idx = 0
            else:
                print(f"❄️/🔥 Dégel partiel : On entraîne les {unfreeze_blocks} derniers blocs sur {total_blocks}.")
                start_idx = total_blocks - unfreeze_blocks

            for i, block in enumerate(blocks):
                if i >= start_idx:
                    for param in block.parameters():
                        param.requires_grad = True
            
            if hasattr(self.dino, 'layernorm'):
                for param in self.dino.layernorm.parameters():
                    param.requires_grad = True
            elif hasattr(self.dino, 'norm'):
                for param in self.dino.norm.parameters():
                    param.requires_grad = True
        else:
            print("❄️ DINOv3 est totalement gelé (Feature Extractor uniquement).")

        # --- 3. HEADS ---
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.head = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

    def _generate_cubemap_coords(self, S):
        rng = np.linspace(-1, 1, S)
        grid_u, grid_v = np.meshgrid(rng, rng) 
        xyz_list = [
            np.stack([np.ones_like(grid_u), -grid_v, -grid_u], axis=-1),
            np.stack([-np.ones_like(grid_u), -grid_v, grid_u], axis=-1),
            np.stack([grid_u, np.ones_like(grid_u), grid_v], axis=-1),
            np.stack([grid_u, -np.ones_like(grid_u), -grid_v], axis=-1),
            np.stack([grid_u, -grid_v, np.ones_like(grid_u)], axis=-1),
            np.stack([-grid_u, -grid_v, -np.ones_like(grid_u)], axis=-1)
        ]
        row1 = np.concatenate(xyz_list[0:3], axis=1) 
        row2 = np.concatenate(xyz_list[3:6], axis=1)
        full_grid_xyz = np.concatenate([row1, row2], axis=0)
        norm = np.linalg.norm(full_grid_xyz, axis=-1, keepdims=True)
        return (full_grid_xyz / norm).reshape(-1, 3), None

    def _compute_vertex_uv(self, sphere_xyz):
        x, y, z = sphere_xyz.T
        abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
        
        is_x = (abs_x >= abs_y) & (abs_x >= abs_z)
        is_y = (abs_y > abs_x) & (abs_y >= abs_z)
        is_z = (abs_z > abs_x) & (abs_z > abs_y)
        
        face_idx = np.zeros(self.N, dtype=int)
        u_loc, v_loc = np.zeros(self.N), np.zeros(self.N)
        
        m = is_x & (x > 0); face_idx[m]=0; u_loc[m]=-z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_x & (x < 0); face_idx[m]=1; u_loc[m]=z[m]/abs_x[m]; v_loc[m]=-y[m]/abs_x[m]
        m = is_y & (y > 0); face_idx[m]=2; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=z[m]/abs_y[m]
        m = is_y & (y < 0); face_idx[m]=3; u_loc[m]=x[m]/abs_y[m]; v_loc[m]=-z[m]/abs_y[m]
        m = is_z & (z > 0); face_idx[m]=4; u_loc[m]=x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]
        m = is_z & (z < 0); face_idx[m]=5; u_loc[m]=-x[m]/abs_z[m]; v_loc[m]=-y[m]/abs_z[m]

        col, row = face_idx % 3, face_idx // 3
        u_px = (u_loc + 1) * 0.5 * (self.face_size - 1)
        v_px = (v_loc + 1) * 0.5 * (self.face_size - 1)
        
        return np.stack([u_px + (col * self.face_size), v_px + (row * self.face_size)], axis=1)

    def project_sphere_to_image(self, x):
        B, C, N = x.shape
        idx = self.pixel_to_vertex_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        flat_img = torch.gather(x, 2, idx) 
        return flat_img.view(B, C, self.H_total, self.W_total)

    def backproject_logits_to_sphere(self, logits_2d):
        B, C, H, W = logits_2d.shape
        u = self.vertex_uv[:, 0].clamp(0, W - 1).long()
        v = self.vertex_uv[:, 1].clamp(0, H - 1).long()
        
        logits_sphere = torch.zeros(B, C, self.N, device=logits_2d.device)
        for b in range(B):
            logits_sphere[b] = logits_2d[b, :, v, u]
        return logits_sphere

    def forward(self, x):
        x_img = self.project_sphere_to_image(x)

        x_rgb = self.input_adapter(x_img)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_rgb - mean) / std

        enable_grad = self.unfreeze_blocks != 0 and self.training
        with torch.set_grad_enabled(enable_grad):
            outputs = self.dino(pixel_values=x_norm)
            hidden_states = outputs.last_hidden_state

        tokens_to_skip = 1 + self.num_registers
        patch_tokens = hidden_states[:, tokens_to_skip:, :]

        B, N_patches, Dim = patch_tokens.shape
        H_grid = self.H_total // self.patch_size
        W_grid = self.W_total // self.patch_size
        
        features_2d = patch_tokens.permute(0, 2, 1).reshape(B, Dim, H_grid, W_grid)
        
        logits_low_res = self.head(features_2d)
        
        logits_high_res = F.interpolate(
            logits_low_res, 
            size=self.img_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        logits_sphere = self.backproject_logits_to_sphere(logits_high_res)
        
        return logits_sphere



if __name__ == "__main__":
    in_order = 7
    in_channels = 1
    out_classes = 69
    batch_size = 2

    n_vertices = number_of_ico_vertices(in_order)
    print (n_vertices)
    x = torch.randn(batch_size, in_channels, n_vertices)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device=",DEVICE)
    print ("start model in general code ")
    s=time.time()
    '''model = MySphericalUNet(
        in_order=in_order,
        in_channels=in_channels,
        out_channels=out_classes,
        base_filters=16,
        depth=3
    ).to(DEVICE)'''

    model2= SphericalUNet(
    in_order= in_order,
    in_channels=in_channels,
    out_channels=out_classes,
    
    depth=3,

    start_filts=4,
    conv_mode="DiNe",
    dine_size=3,
    up_mode="interp",
    standard_ico=False,
    cachedir="/neurospin/dico/stounsi/Runs/spherical_labelling/models/surfify_cache"  
)
    t=time.time()
    print ("temps en mintes", (t-s)/60)
    y = model2(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    assert y.shape == (batch_size, out_classes, n_vertices), "❌ Shape mismatch !"
    print("✅ Test passé : shape correcte.")