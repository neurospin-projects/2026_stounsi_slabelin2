import os
import torch
import numpy as np
import random
from surfify.utils import icosahedron
from utils.figures import plot_3_data_label_preiction, plot_esis_vs_regions
import math

import networkx as nx
from scipy.stats import mode
from scipy.spatial import cKDTree

def build_sphere_graph(faces):
    """Construit le graphe d'adjacence de la sphère à partir des faces (triangles)."""
    G = nx.Graph()
    edges = np.concatenate([
        faces[:, [0, 1]], 
        faces[:, [1, 2]], 
        faces[:, [2, 0]]
    ], axis=0)
    G.add_edges_from(edges)
    return G

def get_nearest_non_zero_neighbor(graph, start_nodes, current_labels, max_depth=3):
    """
    Recherche le label majoritaire des voisins non-nuls dans un rayon de 'max_depth'.
    Utilisé pour franchir le background lors du nettoyage topologique.
    """
    current_frontier = set(start_nodes)
    visited = set(start_nodes)
    
    found_labels = []
    
    for depth in range(max_depth):
        next_frontier = set()
        
        for node in current_frontier:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    lbl = current_labels[neighbor]
                    if lbl != 0:
                        # Voisin valide trouvé (sillon)
                        found_labels.append(lbl)
                    
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        
        if found_labels:
            return mode(found_labels, keepdims=False)[0]
            
        current_frontier = next_frontier
        if not current_frontier:
            break
            
    return None

def clean_small_components(pred_labels, graph, min_size=15):

    cleaned = np.copy(pred_labels)
    active_mask = (pred_labels != 0)
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) == 0: return cleaned

    active_subgraph = graph.subgraph(active_indices)

    # 1. Pré-calcul de la taille des composants pour validation rapide des voisins
    node_comp_sizes = {}
    
    full_color_graph = nx.Graph()
    full_color_graph.add_nodes_from(active_indices)
    for u, v in active_subgraph.edges():
        if pred_labels[u] == pred_labels[v]:
            full_color_graph.add_edge(u, v)
            
    for comp in nx.connected_components(full_color_graph):
        size = len(comp)
        for n in comp:
            node_comp_sizes[n] = size

    # 2. Nettoyage et réassignation
    for comp in nx.connected_components(full_color_graph):
        if len(comp) < min_size:
            nodes_to_clean = list(comp)
            
            valid_neighbors_labels = []
            
            for n in nodes_to_clean:
                for neighbor in graph.neighbors(n):
                    if neighbor in active_indices and neighbor not in comp:
                        
                        # Filtrage : le composant voisin doit dépasser le seuil min_size
                        neighbor_size = node_comp_sizes.get(neighbor, 0)
                        
                        if neighbor_size >= min_size:
                            valid_neighbors_labels.append(cleaned[neighbor])
            
            if valid_neighbors_labels:
                # Assignation du label majoritaire parmi les voisins robustes
                new_label = mode(valid_neighbors_labels, keepdims=False)[0]
                cleaned[nodes_to_clean] = new_label
            else:
                # Fallback : conservation du label initial si aucun voisin robuste n'est trouvé
                pass

    return cleaned

def get_eval_coords(cfg):
    """Charge le template sphérique approprié selon la configuration."""
    if cfg.data.get('zhao_neigh', False):
        from s3pipe.utils.utils import get_sphere_template
        template = get_sphere_template(163842)
        return template["vertices"]
    else:
        coords, faces = icosahedron(order=cfg.data.ico_order, standard_ico=True)
        return coords,faces

def calculate_single_subject_metrics(pred, target, num_classes):
    """
    Calcule les métriques pour un sujet unique.
    Retourne un dictionnaire des scores agrégés.
    """
    # Total des pixels (hors classes ignorées : 0, 2, 4)
    total_sulci_pixels = target.numel() - (target == 0).sum().item() - (target == 2).sum().item() - (target == 4).sum().item()
    if total_sulci_pixels == 0: total_sulci_pixels = 1.0
    sum_inverse_sizes =0
    subject_ious = []
    subject_dices = []
    
    subject_elocal_per_class = np.full(num_classes, np.nan)
    esi_score = 0.0
    inv_esi_score = 0.0
    
    for cls in range(1, num_classes):
        if cls == 2 or cls == 4:
            continue
            
        TP = ((pred == cls) & (target == cls)).sum().item()
        FP = ((pred == cls) & (target != cls)).sum().item()
        FN = ((pred != cls) & (target == cls)).sum().item()
        
        denom_dice = 2 * TP + FP + FN
        
        # --- IoU & Dice ---
        if denom_dice > 0:
            subject_ious.append(TP / (TP + FP + FN))
            subject_dices.append(2 * TP / denom_dice)

        # --- Elocal ---
        if denom_dice > 0:
            elocal = (FP + FN) / ( TP + FP + FN)
            subject_elocal_per_class[cls] = elocal
        else:
            # np.nan permet d'exclure les classes vides des moyennes globales
            subject_elocal_per_class[cls] = np.nan 

        # --- ESI ---
        sl = TP + FN
        if sl > 0:
            sum_inverse_sizes += (1/sl)
            wl = sl / total_sulci_pixels
            term_error = (FP + FN) / denom_dice
            esi_score += wl * term_error
            inv_esi_score += (1/sl) * term_error
           
  

    return {
        "iou": np.mean(subject_ious) if subject_ious else 0.0,
        "dice": np.mean(subject_dices) if subject_dices else 0.0,
        "elocal_vector": subject_elocal_per_class, 
        "esi": esi_score , 
        "invesi": inv_esi_score/(sum_inverse_sizes)

    }

def evaluate(model, val_loader, cfg, device, epoch, plot=True, save_dir=None,criterion=None):
    model.eval()
    coords, faces = get_eval_coords(cfg)
    num_classes = cfg.data.num_classes

    if isinstance(faces, torch.Tensor):
        faces_np = faces.cpu().numpy()
    else:
        faces_np = faces
    
    if cfg.training.post_train : 
        sphere_graph = build_sphere_graph(faces_np)
    
    patient_metrics = {
        "iou": [],
        "dice": [],
        #"elocal": [], 
        "all_elocals_matrix" : [],
        "esi": [] , 
        "invesi" : []
    }
    
    total_correct = 0
    total_pixels = 0
    total_val_loss = 0.0

    with torch.no_grad():
        for i, (c, f, X, Y) in enumerate(val_loader):
            X = X.to(device, dtype=torch.float32)
            Y = Y.to(device)
            
            if not cfg.data.use_lines:
                model_input = X[:, 1:, :] 
            else:
                model_input = X 
            
            logits = model(model_input)
            if criterion is not None:
                loss = criterion(logits, Y)
                total_val_loss += loss.item()
            preds = logits.argmax(dim=1)
            preds1 = preds * X[:, 0, :] # Masquage du fond

            preds_np = preds1.cpu().numpy()
            if cfg.training.post_train : 
                
                cleaned_preds_list = []
                # Itération par sujet pour le post-processing topologique
                for b in range(preds_np.shape[0]):
                    p_clean = clean_small_components(preds_np[b], sphere_graph, min_size=10)
                    cleaned_preds_list.append(p_clean)
                
                preds = torch.tensor(np.stack(cleaned_preds_list), device=device)
            else : 
                preds= preds1
            
            total_correct += (preds == Y).sum().item()
            total_pixels += Y.numel()

            # Extraction des métriques par sujet
            batch_size = X.shape[0]
            for b in range(batch_size):
                metrics_b = calculate_single_subject_metrics(preds[b], Y[b], num_classes)
                
                patient_metrics["iou"].append(metrics_b["iou"])
                patient_metrics["dice"].append(metrics_b["dice"])
                #patient_metrics["elocal"].append(metrics_b["elocal"]) 
                patient_metrics["esi"].append(metrics_b["esi"])
                patient_metrics["invesi"].append(metrics_b["invesi"])
                patient_metrics["all_elocals_matrix"].append(metrics_b["elocal_vector"])

            if plot and i == 0 and save_dir: 
                idx_in_batch = random.randint(0, X.shape[0] - 1)
                
                fig_name = f"prediction_plot_epoch_{epoch}.png"
                save_path = os.path.join(save_dir, fig_name)
                
                plot_3_data_label_preiction(
                    coords,
#                    faces,
                    # X, preds, Y, 
                    Y.unsqueeze(1),preds1 , preds,
                    
                    save_path, 
                    i=idx_in_batch, 
                    num_classes=num_classes
                )
                
    mean_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    if patient_metrics["all_elocals_matrix"]:
        elocal_matrix = np.vstack(patient_metrics["all_elocals_matrix"])
    else:
        elocal_matrix = np.zeros((0, cfg.data.num_classes))
        
    results = {
        "val_loss": mean_val_loss,
        "mean_iou": np.mean(patient_metrics["iou"]) if patient_metrics["iou"] else 0.0,
        "mean_dice": np.mean(patient_metrics["dice"]) if patient_metrics["dice"] else 0.0,
        #"mean_elocal": np.mean(patient_metrics["elocal"]) if patient_metrics["elocal"] else 0.0,
        "esi": np.mean(patient_metrics["esi"]) if patient_metrics["esi"] else 0.0,
        "invesi": np.mean(patient_metrics["invesi"]) if patient_metrics["invesi"] else 0.0,
        "pixel_accuracy": total_correct / total_pixels if total_pixels > 0 else 0.0,
        
        "per_subject_iou": patient_metrics["iou"],
        "per_subject_dice": patient_metrics["dice"],
        #"per_subject_elocal": patient_metrics["elocal"], 
        "elocal_matrix": elocal_matrix,
        "per_subject_esi": patient_metrics["esi"],
        "per_subject_invesi": patient_metrics["invesi"]
    }
    
    return results








# def evaluate(model, val_loader, NUM_CLASSES, DEVICE,epoch,plot=True,save_file=None,zhao=False,hemisphere="lh"):

#     if zhao : 
#         from s3pipe.utils.utils import get_sphere_template

#         template = get_sphere_template(163842)  
#         coords = template["vertices"]
#     else : 
#         coords,faces=icosahedron(order=7, standard_ico=True)
#     model.eval()
#     total_correct = 0
#     total_pixels = 0
#     iou_scores = []
#     elocal_scores = []

#     esis=[]
#     sulci_length=[]
#     esi_per_sulci=[]

#     with torch.no_grad():
#         for X, Y in val_loader:
#             X = X.to(DEVICE)  # (B, 1, N)
#             X=X.to(dtype=torch.float32, device=DEVICE)
#             Y = Y.to(DEVICE)  # (B, N)
#             logits = model(X[:,1,:].unsqueeze(1))  # (B, C, N)
#             preds = logits.argmax(dim=1)  # (B, N)
#             preds=preds*X[:,0,:]
            

#             '''# Simule des prédictions où la moitié des premiers labels sont corrects
#             B, N = Y.shape
#             preds = torch.empty_like(Y)

#             for b in range(B):
#                 n_correct = N // 2
#                 idx = torch.randperm(N)
#                 correct_idx = idx[:n_correct]
#                 wrong_idx = idx[n_correct:]

#                 # Mettre les bonnes prédictions
#                 preds[b, correct_idx] = Y[b, correct_idx]

#                 # Faux labels aléatoires différents de Y
#                 rand_wrong = torch.randint(1, NUM_CLASSES, size=(len(wrong_idx),), device=Y.device)
#                 # s'assurer que le faux label ≠ vrai label
#                 for i, j in enumerate(wrong_idx):
#                     while rand_wrong[i] == Y[b, j]:
#                         rand_wrong[i] = torch.randint(1, NUM_CLASSES, (1,), device=Y.device)
#                 preds[b, wrong_idx] = rand_wrong'''

            




        

#             total_correct += (preds == Y).sum().item()
#             total_pixels += Y.numel()
#             total_sulci=Y.numel()- (Y==0).sum().item() - (Y==2).sum().item()- (Y==14).sum().item()
        
#             esi_sub=0
#             for cls in range(1, NUM_CLASSES):  # ignore background class 0
#                 if cls!=2 and cls!=14:
#                     TP = ((preds == cls) & (Y == cls)).sum().item()
#                     FP = ((preds == cls) & (Y != cls)).sum().item()
#                     FN = ((preds != cls) & (Y == cls)).sum().item()

#                     union = TP + FP + FN

#                     # IoU
#                     if TP + FP + FN > 0:
#                         iou_scores.append(TP / (TP + FP + FN))

#                     # Elocal (par sulcus)
#                     if TP + FP + FN > 0:
#                         elocal = (FP + FN) / (TP + FP + FN)
#                         elocal_scores.append(elocal)
#                     else : 
#                         # cas extrême (présent dans prédiction ou dans vérité mais pas l'autre)
#                         elocal_scores.append(1.0)

#                     # ESI composants
#                     sl = TP + FN
#                     sulci_length.append(sl)
#                     if sl > 0:
#                         wl = sl / total_sulci 
#                         esi_sub += wl * (FP + FN) /(2 * TP + FP + FN)
#                         esi_per_sulci.append(esi_sub)
#             esis.append(esi_sub)
                    

#     pixel_accuracy = total_correct / total_pixels
#     mean_iou = np.mean(iou_scores) if iou_scores else 0.0
#     mean_elocal = np.mean(elocal_scores) if elocal_scores else 0.0
#     #print (len(elocal_scores))
#     esi = np.mean(esis) if esis else 0.0
#     if plot :
#         num=X.shape[0]
#         #print ("num",num)
#         plot_3_data_label_preiction(coords, X,preds,Y,save_file,i=random.randint(0,num-1),num_classes=NUM_CLASSES)
#         new_save_file = save_file.replace("prediction_plot", "esi_per_sulci_difference")
#         elocal_scores = np.array(elocal_scores).reshape(len(elocal_scores)//(NUM_CLASSES-3), NUM_CLASSES-3).mean(axis=0).tolist()
#         sulci_length= np.array(sulci_length).reshape(len(sulci_length)//(NUM_CLASSES-3), NUM_CLASSES-3).mean(axis=0).tolist()
     
        
#         #plot_esis_vs_regions("/home/st283990/pclean_freesurfer/matrice_correspondance2_"+hemisphere+".txt", elocal_scores, sulci_length, new_save_file)
#         #plot_esis_vs_regions("/lustre/fsn1/projects/rech/tgu/uxc45lm/pclean_freesurfer/matrice_correspondance2_"+hemisphere+".txt", elocal_scores, sulci_length, new_save_file)
#         #difference_avec_leonie("/home/st283990/pclean_freesurfer/matrice_correspondance2_"+hemisphere+".txt", elocal_scores, sulci_length, new_save_file, hemisphere)

#     return pixel_accuracy, mean_iou, mean_elocal, esi