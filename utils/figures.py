import torch 
import matplotlib.pyplot as plt

def plot_occurancy_from_logits (logits, save_file) : 
    preds = logits.argmax(dim=1)

    plt.figure(figsize=(12, 3))
    plt.plot(preds[0,0:80000].cpu().numpy(), linewidth=0.1)
    plt.xlabel("Index")
    plt.ylabel("Valeur binaire")
    plt.title("Masque binaire (plot)")
    plt.tight_layout()
    plt.show()
    plt.savefig(save_file)





import numpy as np

from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_multi_lab_sphere(coords, logits,save_file,num_classes=67,i=0,preds=False,save=True) : 
    mask_transfer = logits.argmax(dim=1)
    if preds : 
        mask_transfer = logits

    
    classes_i = mask_transfer[i].cpu().numpy()  # (N,)
    coords_np = coords.cpu().numpy()  # (N, 3)
    

    # Colormap catégorielle : HSV avec couleur personnalisée pour la classe 0 (fond)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes))
    colors[0] = [0.95,0.80,0.50, 0.8]

    cmap = ListedColormap(colors)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, num_classes)

    # Rendu 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        coords_np[:, 0], coords_np[:, 1], coords_np[:, 2],
        c=classes_i, cmap=cmap, norm=norm, s=1
    )

    tick_step = 1
    cb = plt.colorbar(sc, ax=ax, ticks=np.arange(0, num_classes, tick_step), fraction=0.02, pad=0.05)
    cb.set_label("Classe")

    ax.set_title(f"Visualisation 3D des classes (échantillon {i})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save : 
        plt.savefig(save_file)
    plt.show()
   


def plot_binary_lab_sphere_(coords, mask_output,save_file,i=0,save=True ) : 

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=mask_output.cpu().numpy()[i,0,:], cmap='YlGnBu')

    plt.colorbar(sc, ax=ax, label="Valeur scalaire")
    ax.set_title("Visualisation de la sphère avec valeurs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if save : 
        plt.savefig(save_file)
    plt.show()
    plt.close()

    
 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
# Import nécessaire pour la 3D
from mpl_toolkits.mplot3d import Axes3D 
import pyvista as pv
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Variante PyVista (Fond Noir)
# def plot_3_data_label_preiction(coords, mask_output, mask_transfer, preds, save_file, i=0, num_classes=67):
    
#     # 1. Préparation des données
#     coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else coords
    
#     labels_pred = mask_transfer[i].cpu().numpy() if hasattr(mask_transfer, 'cpu') else mask_transfer[i]
#     labels_gt = preds[i].cpu().numpy() if hasattr(preds, 'cpu') else preds[i]
#     labels_binary = mask_output.cpu().numpy()[i, 0, :] if hasattr(mask_output, 'cpu') else mask_output[i, 0, :]

#     data_list = [labels_pred, labels_gt, labels_binary]
#     titles = [f"Prédictions (Sample {i})", f"Vérité Terrain (Sample {i})", f"Input Binaire (Sample {i})"]

#     # 2. Configuration du Plotter PyVista
#     pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 800))
#     pl.set_background('black')

#     # Sphère d'occultation centrale
#     sphere_radius = 0.98 
    
#     # 3. Rendu par sous-fenêtre
#     for idx, labels in enumerate(data_list):
#         pl.subplot(0, idx)
#         pl.add_text(titles[idx], color='white', font_size=18)

#         # Filtrage des points d'arrière-plan (classe 0)
#         mask_active = labels > 0 
        
#         if np.sum(mask_active) == 0:
#             continue 
            
#         points_active = coords_np[mask_active]
#         labels_active = labels[mask_active]
        
#         cloud = pv.PolyData(points_active)
        
#         # Ajout de la sphère centrale pour l'occultation des faces arrière
#         center_sphere = pv.Sphere(radius=sphere_radius, theta_resolution=60, phi_resolution=60)
#         pl.add_mesh(center_sphere, color='#1a1a1a', ambient=0.2, specular=0.1)

#         if idx == 2: # Input Binaire
#             pl.add_mesh(cloud, 
#                         color='cyan',
#                         render_points_as_spheres=True, 
#                         point_size=2,                 
#                         ambient=0.5,                   
#                         diffuse=0.5)
#         else: # Multi-classes
#             cloud.point_data['classes'] = labels_active
            
#             pl.add_mesh(cloud, 
#                         scalars='classes',
#                         cmap='turbo',
#                         render_points_as_spheres=True, 
#                         point_size=4,
#                         clim=[0, num_classes],         
#                         show_scalar_bar=False,         
#                         ambient=0.3)

#         # Paramétrage de la caméra
#         pl.camera_position = 'yz' 
#         pl.camera.azimuth = 45
#         pl.camera.elevation = 20
#         pl.camera.zoom(1.2) 

#     # 4. Sauvegarde
#     pl.screenshot(save_file)
#     pl.close()
#     del pl






def plot_3_data_label_preiction(coords, mask_output, mask_transfer, preds, save_file, i=0, num_classes=67):
    
    # Paramétrage de la vue
    elev, azim = 120, 45
    
    # 1. Préparation des données
    coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else coords
    
    labels_pred = mask_transfer[i].cpu().numpy() if hasattr(mask_transfer, 'cpu') else mask_transfer[i]
    labels_gt = preds[i].cpu().numpy() if hasattr(preds, 'cpu') else preds[i]
    labels_binary = mask_output.cpu().numpy()[i, 0, :] if hasattr(mask_output, 'cpu') else mask_output[i, 0, :]

    # Filtrage de visibilité (Back-face culling manuel via produit scalaire)
    rad_elev = np.radians(elev)
    rad_azim = np.radians(azim)
    
    cam_x = np.cos(rad_elev) * np.cos(rad_azim)
    cam_y = np.cos(rad_elev) * np.sin(rad_azim)
    cam_z = np.sin(rad_elev)
    
    dot_prod = (coords_np[:, 0] * cam_x) + (coords_np[:, 1] * cam_y) + (coords_np[:, 2] * cam_z)
    
    # Conservation des points orientés vers la caméra (tolérance de -0.2 aux bords)
    visible_mask = dot_prod > -0.2
    
    coords_vis = coords_np[visible_mask]
    labels_pred = labels_pred[visible_mask]
    labels_gt = labels_gt[visible_mask]
    labels_binary = labels_binary[visible_mask]
    
    # 2. Configuration graphique
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 8))
    
    titles = [f"Prédictions (Sample {i})", f"Vérité Terrain (Sample {i})", f"Input Binaire (Sample {i})"]
    data_list = [labels_pred, labels_gt, labels_binary]
    
    # 3. Rendu des sous-graphiques
    for idx, ax_data in enumerate(data_list):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_title(titles[idx], color='white', fontsize=16, pad=10)
        
        if idx == 3: # Input Binaire
            is_line = ax_data > 0.5
            
            ax.scatter(coords_vis[~is_line, 0], coords_vis[~is_line, 1], coords_vis[~is_line, 2],
                       c='#222222', s=0.5, alpha=0.3) 
            
            ax.scatter(coords_vis[is_line, 0], coords_vis[is_line, 1], coords_vis[is_line, 2],
                       c='cyan', s=3, alpha=1.0)
            
        else: # Multi-classes
            cmap = plt.get_cmap('turbo', num_classes)
            is_background = (ax_data == 0)
            
            ax.scatter(coords_vis[is_background, 0], coords_vis[is_background, 1], coords_vis[is_background, 2],
                       c='#1a1a1a', s=0.5, alpha=0.3)
            
            p = ax.scatter(coords_vis[~is_background, 0], coords_vis[~is_background, 1], coords_vis[~is_background, 2],
                       c=ax_data[~is_background], cmap=cmap, s=3, alpha=1.0, 
                       vmin=0, vmax=num_classes)
            
            cbar = plt.colorbar(p, ax=ax, fraction=0.02, pad=0.05)
            cbar.ax.tick_params(labelsize=8, colors='white')
            cbar.outline.set_visible(False)

        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim) 

    plt.tight_layout()
    plt.savefig(save_file, dpi=200, facecolor='black', bbox_inches='tight')
    plt.close()


# Variante PyVista (Fond Blanc)

# def plot_3_data_label_preiction(coords, mask_output, mask_transfer, preds, save_file, i=0, num_classes=67):
#     # 1. Préparation des données
#     cloud = pv.PolyData(coords) 

#     labels_pred = mask_transfer[i].cpu().numpy().flatten()
#     labels_gt = preds[i].cpu().numpy().flatten()
#     scalar_vals = mask_output.cpu().numpy()[i, 0, :].flatten()

#     # Mapping des couleurs (Matplotlib vers PyVista HEX)
#     cmap_base = plt.cm.get_cmap("nipy_spectral", num_classes)
    
#     list_colors = [to_hex(cmap_base(x)) for x in np.linspace(0, 1, num_classes)]
    
#     np.random.seed(42)
#     np.random.shuffle(list_colors)
    
#     list_colors[0] = "#f0f0f0" 

#     # Configuration du Plotter
#     pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2400, 800])
#     pl.enable_anti_aliasing('msaa') 
    
#     pl.subplot(0, 0)
#     pl.add_text(f"Predictions", font_size=10)
#     cloud_pred = cloud.copy()
#     cloud_pred["labels"] = labels_pred
    
#     pl.add_mesh(cloud_pred, scalars="labels", cmap=list_colors, 
#                 point_size=10, render_points_as_spheres=True,
#                 show_scalar_bar=False)
#     pl.camera_position = 'yz' 

#     pl.subplot(0, 1)
#     pl.add_text(f"Verite Terrain", font_size=10)
#     cloud_gt = cloud.copy()
#     cloud_gt["labels"] = labels_gt
    
#     pl.add_mesh(cloud_gt, scalars="labels", cmap=list_colors, 
#                 point_size=10, render_points_as_spheres=True,
#                 show_scalar_bar=False)
#     pl.camera_position = 'yz'

#     pl.subplot(0, 2)
#     pl.add_text(f"Valeurs", font_size=10)
#     cloud_scalar = cloud.copy()
#     cloud_scalar["values"] = scalar_vals
    
#     pl.add_mesh(cloud_scalar, scalars="values", cmap="magma", 
#                 point_size=10, render_points_as_spheres=True,
#                 show_scalar_bar=True)
#     pl.camera_position = 'yz'

#     # Sauvegarde
#     pl.screenshot(save_file)
#     pl.close()


# Variante Nilearn
from nilearn import plotting

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting

# def plot_3_data_label_preiction(coords, faces, mask_output, mask_transfer, preds, save_file, i=0, num_classes=67):
#     """
#     Affiche les données sur une surface avec Nilearn.
    
#     Args:
#         coords (Tensor/Array): Sommets (N, 3).
#         faces (Tensor/Array): Triangles (M, 3).
#         mask_output, mask_transfer, preds: Tenseurs de données (Batch).
#         save_file (str): Chemin de sauvegarde.
#     """
    
#     # 1. Préparation du maillage
#     coords_np = coords.cpu().numpy() if hasattr(coords, 'cpu') else coords
#     faces_np = faces.cpu().numpy() if hasattr(faces, 'cpu') else faces
    
#     mesh = [coords_np, faces_np]

#     labels_pred = mask_transfer[i].cpu().numpy() if hasattr(mask_transfer, 'cpu') else mask_transfer[i]
#     labels_gt = preds[i].cpu().numpy() if hasattr(preds, 'cpu') else preds[i]
#     labels_binary = mask_output.cpu().numpy()[i, 0, :] if hasattr(mask_output, 'cpu') else mask_output[i, 0, :]

#     # 2. Configuration de l'affichage
#     plt.style.use('dark_background')
#     fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})
#     fig.patch.set_facecolor('black') 

#     titles = [f"Prédictions (Sample {i})", f"Vérité Terrain (Sample {i})", f"Input Binaire (Sample {i})"]
#     data_list = [labels_pred, labels_gt, labels_binary]
    
#     view_angles = [20, 45]

#     cmap_binary = ListedColormap(np.array([[0, 0, 0, 0], [0, 1, 1, 1]])) 
#     cmap_multi = plt.get_cmap('turbo', num_classes).copy()
#     cmap_multi.set_under('black') 
    
#     # 3. Rendu via Nilearn
#     for idx, ax in enumerate(axes):
#         data = data_list[idx]
        
#         if idx == 2: 
#             cmap = cmap_binary
#             threshold = 0.5 
#             vmin, vmax = 0, 1
#         else: 
#             cmap = cmap_multi
#             threshold = 0.1 
#             vmin, vmax = 0, num_classes

#         plotting.plot_surf_roi(
#             surf_mesh=mesh, 
#             roi_map=data,
#             hemi='left',       
#             view=view_angles,
#             bg_map=None,       
#             bg_on_data=False,  
#             alpha=1.0,
#             cmap=cmap,
#             vmin=vmin, 
#             vmax=vmax,
#             threshold=threshold,
#             axes=ax,
#             figure=fig
#         )
        
#         ax.set_title(titles[idx], color='white', fontsize=18, pad=20)
#         ax.set_facecolor('black')
        
#         ax.axis('off')

#     # 4. Sauvegarde
#     plt.tight_layout()
#     plt.savefig(save_file, dpi=200, facecolor='black', bbox_inches='tight')
#     plt.close()





def plot_esis_vs_regions(matrice_file, esis, sulci_length, save_file):
    """
    Crée et enregistre un graphique ESIS trié selon sulci_length.

    Args:
        matrice_file (str): Chemin vers le fichier matrice.txt
        esis (list of float): Liste des scores ESIS
        sulci_length (list of float): Liste des longueurs de sillons
        save_file (str): Chemin pour sauvegarder le graphique
    """
    matrice = pd.read_csv(matrice_file, sep="\t", header=None, names=["Region", "Index"])
    
    # Filtrage des indices ignorés (2 et 14)
    matrice = matrice[~matrice["Index"].isin([2, 14])].reset_index(drop=True)



    # Formatage des labels de régions
    matrice["Region"] = matrice["Region"].str.replace("_", " ")

    matrice["Score_ESIS"] = esis
    matrice["Sulci_Length"] = sulci_length

    # Tri décroissant par longueur de sillon
    matrice = matrice.sort_values("Sulci_Length",ascending=False)

    # Rendu du graphique barh
    plt.figure(figsize=(14, 18))
    bars=plt.barh(matrice["Region"], matrice["Score_ESIS"], color='teal')
    
    plt.xlabel("Score Elocal ", fontsize=12)
    plt.ylabel("Sulci", fontsize=12)
    plt.title("ELocal (sort by sulci lenghth)", fontsize=14)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(width , y, f"{width:.2f}", fontsize=9, color='black')
    
    plt.savefig(save_file)
    plt.close() 

    print(f"✅ Graphique enregistré dans : {save_file}")


def difference_avec_leonie(matrice_file, esis, sulci_length, save_file, hemisphere='lh'):
    """
    Crée et enregistre un graphique comparatif des différences de performance ESIS.

    Args:
        matrice_file (str): Chemin vers le fichier matrice.txt
        esis (list of float): Liste des scores ESIS
        sulci_length (list of float): Liste des longueurs de sillons
        save_file (str): Chemin pour sauvegarder le graphique
        hemisphere (str): L'hémisphère à traiter ('lh' ou 'rh').
    """
    matrice = pd.read_csv(matrice_file, sep="\t", header=None, names=["Region", "Index"])
    
    # Filtrage des indices ignorés
    matrice = matrice[~matrice["Index"].isin([2, 14])].reset_index(drop=True)



    matrice["Region"] = matrice["Region"].str.replace("_", " ")

    matrice["Score_ESIS"] = esis
    matrice["Sulci_Length"] = sulci_length

    matrice = matrice.sort_values("Sulci_Length",ascending=False)
    df = pd.read_csv("/home/st283990/pclean_freesurfer/results_leonie_elocal.txt", sep=",")

    # Mapping spécifique par hémisphère
    if hemisphere == 'lh':
        hemi_suffix = " left"
        spam_col = "Mean Local E Left SPAM"
        unet_col = "Mean Local E Left UNET"
        max_spam_col = "Max Local E Left SPAM"
        max_unet_col = "Max Local E Left UNET"
    elif hemisphere == 'rh':
        hemi_suffix = " right"
        spam_col = "Mean Local E Right SPAM"
        unet_col = "Mean Local E Right UNET"
        max_spam_col = "Max Local E Right SPAM"
        max_unet_col = "Max Local E Right UNET"
    else:
        raise ValueError("hemisphere must be 'lh' or 'rh'")

    df_hemi = df[["Region", spam_col, unet_col, max_spam_col, max_unet_col]].copy()

    # Alignement de la nomenclature des régions
    df_hemi["Region"] = df_hemi["Region"].astype(str) + hemi_suffix

    # Jointure et calcul des deltas
    merged = matrice.merge(df_hemi, on="Region", how="inner")
    merged["diff"] = merged["Score_ESIS"] - merged[unet_col]/100

    # Rendu du graphique
    plt.figure(figsize=(14, 18))
    bars=plt.barh(merged["Region"], merged["diff"], color='teal')
    
    plt.xlabel("Score Elocal ", fontsize=12)
    plt.ylabel("Sulci", fontsize=12)
    plt.title(f"Différence de performance Elocal par sillon ({hemisphere.upper()})", fontsize=14)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        plt.text(width , y, f"{width:.2f}", fontsize=9, color='black')
    
    plt.savefig(save_file)
    plt.close() 

    print(f"✅ Graphique enregistré dans : {save_file}")