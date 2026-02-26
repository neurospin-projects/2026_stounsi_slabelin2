# training/trainer.py
import os
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from evaluation.eval import evaluate

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0, mode='min'):
        """
        Args:
            patience (int): Nombre d'époques tolérées sans amélioration.
            min_delta (float): Variation minimale requise pour acter une amélioration.
            mode (str): 'min' pour minimiser (ex: loss), 'max' pour maximiser (ex: IoU).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Configuration de l'opérateur logique selon la métrique ciblée
        if mode == 'min':
            self.monitor_op = np.less  
            self.delta_op = lambda a, b: a - b 
        elif mode == 'max':
            self.monitor_op = np.greater 
            self.delta_op = lambda a, b: b - a 
        else:
            raise ValueError(f"Mode {mode} inconnu. Utiliser 'min' ou 'max'.")

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        
        # Vérification du seuil d'amélioration
        elif self.delta_op(self.best_score, current_score) < self.min_delta:
            self.counter += 1
          
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0 

class Trainer:
    def __init__(self, cfg, model, optimizer, criterion, train_loader, val_loader,fold_save=False,fold=0,start_epoch=0, best_iou=0.0):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(cfg.device)
        self.fold_save=fold_save
        self.fold=fold
        
        self.figures_dir = os.path.join(cfg.paths.output_dir, "figures")
        self.logs_dir = os.path.join(cfg.paths.output_dir, "logs")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.logs_dir)
        self.scaler = GradScaler(self.device.type)
        self.best_iou = best_iou       
        self.start_epoch = start_epoch 

    def print_adapter_weights(self, epoch):
        if hasattr(self.model, 'input_adapter'):
            # Extraction des poids (Shape: [Out_Channels=2, In_Channels=3, Kernel=1])
            # Indices : 0=Ligne, 1=Curv, 2=Sulc
            w = self.model.input_adapter.weight.detach().cpu()
            
            out0_ligne = w[0, 0, 0].item()
            out0_curv  = w[0, 1, 0].item()
            out0_sulc  = w[0, 2, 0].item()
            
            out1_ligne = w[1, 0, 0].item()
            out1_curv  = w[1, 1, 0].item()
            out1_sulc  = w[1, 2, 0].item()

            # print(f"\n🔍 [Epoch {epoch}] Mixage Adaptateur (Entrée -> Sortie) :")
            # print(f"   ➤ OUT 0 (Base Curv) : {out0_ligne:.3f}*Ligne + {out0_curv:.3f}*Curv + {out0_sulc:.3f}*Sulc")
            # print(f"   ➤ OUT 1 (Base Sulc) : {out1_ligne:.3f}*Ligne + {out1_curv:.3f}*Curv + {out1_sulc:.3f}*Sulc")

            # Suivi TensorBoard des poids liés aux lignes
            self.writer.add_scalar("Adapter/Out0_Weight_Ligne", out0_ligne, epoch)
            self.writer.add_scalar("Adapter/Out1_Weight_Ligne", out1_ligne, epoch)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        for i, (coords, faces, X, Y) in enumerate(self.train_loader):
            X = X.to(dtype=torch.float32, device=self.device)
           
            Y = Y.to(self.device)
            
            # Filtrage du canal des lignes (Channel 0) selon la configuration
            if not self.cfg.data.use_lines:
                model_input = X[:, 1:, :] 
            else:
                model_input = X 
                
            self.optimizer.zero_grad()
            
            with autocast(self.device.type):
                logits = self.model(model_input) 
                loss = self.criterion(logits, Y)

            if not torch.isfinite(loss):
                print(f"❌ Loss non-finite à l'epoch {epoch}!")
                return None 

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar(f"Loss/train", avg_loss, epoch)
        
        if (epoch + 1) % self.cfg.training.log_freq == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            self.print_adapter_weights(epoch + 1)

    def validate(self, epoch):
        save_file = os.path.join(self.figures_dir, f"prediction_plot_{epoch}.png")
        plot = (epoch + 1) % self.cfg.training.save_freq == 0
        
        metrics = evaluate(
    model=self.model,
    val_loader=self.val_loader,
    cfg=self.cfg,
    device=self.device,
    epoch=epoch,
    plot=plot,
    save_dir=self.figures_dir,
    criterion=self.criterion
)
        #print(f"Validation IoU: {metrics['mean_iou']:.4f}")
        self.writer.add_scalar("IoU/val", metrics['mean_iou'], epoch)
        self.writer.add_scalar("ESI/val", metrics['esi'], epoch)
        self.writer.add_scalar("INV_ESI/val", metrics['invesi'], epoch)
        self.writer.add_scalar("DICE/val", metrics['mean_dice'], epoch)
        self.writer.add_scalar("Loss/val", metrics['val_loss'], epoch)

        
        #print(f"[Val Epoch {epoch+1}] IoU: {metrics['mean_iou']:.4f} (Best: {self.best_iou:.4f})")

        if metrics['mean_iou'] > self.best_iou:
            if self.fold_save: 
                fold_output_dir = os.path.join(self.cfg.paths.output_dir, f"fold_{self.fold}")
                os.makedirs(fold_output_dir, exist_ok=True)
            self.best_iou = metrics['mean_iou']
        
            self._save_checkpoint(epoch, metrics['mean_iou'])
        return metrics

    def _save_checkpoint(self, epoch, iou):
        if self.fold_save : 
        
            path = os.path.join(self.cfg.paths.output_dir, f"fold_{self.fold}", "best_model.pth")
        else : 
            path = os.path.join(self.cfg.paths.output_dir, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': iou,
            'config': self.cfg
        }, path)
        print(f"✅ Modèle sauvegardé : {path}")

    def run(self):
        print(f"🚀 Démarrage entrainement : {self.cfg.experiment_name}")
        
        # Initialisation du protocole d'arrêt anticipé
        patience = self.cfg.training.get('early_stopping_patience', 30)
        mode = self.cfg.training.get('monitor_mode', 'min') 
        metric_name = self.cfg.training.get('monitor_metric', 'loss')
        min_delta=self.cfg.training.get('early_stopping_min_delta',0.01)
        
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode=mode)
        print(f"👀 Surveillance pour arrêt anticipé : {metric_name} (Mode: {mode})")

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            self.train_epoch(epoch)
            
            if (epoch + 1) % self.cfg.training.val_freq == 0:
                metrics = self.validate(epoch) 
                
                # Routage de la métrique ciblée
                if metric_name == 'loss':
                    current_score = metrics['val_loss']
                elif metric_name == 'dice':
                    current_score = metrics['mean_dice']
                elif metric_name == 'iou':
                    current_score = metrics['mean_iou']
                else:
                    current_score = metrics['val_loss']

                early_stopper(current_score)
                
                if early_stopper.early_stop:
                    print(f"\n🛑 Arrêt anticipé à l'epoch {epoch+1} !")
                    print(f"   Le {metric_name} ne s'améliore plus depuis {patience} validations.")
                    break
        
        self.writer.close()