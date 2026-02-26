import torch
import torch.nn as nn
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
from omegaconf import DictConfig

try:
    from surfify.models.base import SphericalBase
    from surfify.nn import IcoPool
except ImportError:
    print("⚠️ Attention: surfify manquant, impossible d'instancier le vrai modèle.")
    SphericalBase = object


class SingleHemiVGG16BN(SphericalBase):
    """
    VGG Modulable et Autonome pour SSL.
    Gère dynamiquement :
     - La profondeur (num_blocks)
     - La largeur (start_channels)
     - Le type de pooling (max, mean ou flatten)
    """
    def __init__(self, input_channels: int, output_dim: int, params: DictConfig):
        
        self.input_order = params.get("input_order", 7)
        self.start_channels = params.get("start_channels", 32)
        self.num_blocks = params.get("num_blocks", 4)
        
        self.pool_type = params.get("pooling_type", "max") 
        
        # Type de sortie finale
        self.global_pool_type = params.get("global_pooling", "mean") 

        super().__init__(
            input_order=self.input_order, 
            n_layers=self.num_blocks, 
            conv_mode=params.get("conv_mode", "DiNe"), 
            dine_size=params.get("dine_size", 1),
            repa_size=params.get("repa_size", 5),
            repa_zoom=params.get("repa_zoom", 5),
            dynamic_repa_zoom=params.get("dynamic_repa_zoom", False),
            standard_ico=params.get("standard_ico", True),
            cachedir=params.get("cachedir", None)
        )
        
        log.info(f"Initialisation SingleHemiVGG16BN Autonome")
        log.info(f" -> IcoPool: {self.pool_type} | Final Aggregation: {self.global_pool_type}")

        self.encoder_layers = nn.ModuleList()
        
        current_channels = input_channels
        block_channels = self.start_channels
        current_order = self.input_order

        for i in range(self.num_blocks):
            self.encoder_layers.append(self.sconv(current_channels, block_channels, self.ico[current_order].conv_neighbor_indices))
            self.encoder_layers.append(nn.BatchNorm1d(block_channels))
            self.encoder_layers.append(nn.ReLU(inplace=True))
            
            self.encoder_layers.append(self.sconv(block_channels, block_channels, self.ico[current_order].conv_neighbor_indices))
            self.encoder_layers.append(nn.BatchNorm1d(block_channels))
            self.encoder_layers.append(nn.ReLU(inplace=True))
            
            current_order -= 1
            pooling = IcoPool(
                down_neigh_indices=self.ico[current_order + 1].neighbor_indices,
                down_indices=self.ico[current_order + 1].down_indices,
                pooling_type=self.pool_type 
            )
            self.encoder_layers.append(pooling)
            
            current_channels = block_channels
            
            if i < self.num_blocks - 1:
                block_channels *= 2

        self.final_conv_channels = current_channels
        
        if self.global_pool_type == "flatten":
            final_order = self.input_order - self.num_blocks
            self.final_n_vertices = 10 * (4 ** final_order) + 2
            
            flat_dim = self.final_conv_channels * self.final_n_vertices
            
            log.info(f"Mode Flatten activé. Output Order: {final_order} ({self.final_n_vertices} pts). Flatten Dim: {flat_dim}")
            
            self.final_proj = nn.Linear(flat_dim, output_dim)
            
        else:
            if self.final_conv_channels != output_dim:
                self.final_proj = nn.Linear(self.final_conv_channels, output_dim)
            else:
                self.final_proj = nn.Identity()
        self.final_bn = nn.BatchNorm1d(self.final_conv_channels)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
            if isinstance(x, tuple): 
                x = x[0]

        if self.global_pool_type == "flatten":
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            
        elif self.global_pool_type == "max":
            x = x.max(dim=-1)[0]
            
        else:
            x = x.mean(dim=-1)
        
        #x = self.final_bn(x)
        x = self.final_proj(x)
        
        return x




def load_simple_model(ckpt_path):
    print(f"📂 Chargement de : {ckpt_path}")
    
    # A. Configuration manuelle
    my_params = {
        "input_order": 6,
        "start_channels": 32,
        "num_blocks": 5,
        "pooling_type": "max",
        "global_pooling": "flatten",
        "conv_mode": "DiNe",
                 
        "dine_size": 1 ,              
        "repa_size": 5  ,             
        "repa_zoom": 5   ,            
        "dynamic_repa_zoom": False  , 
        "standard_ico": True    ,     
        "cachedir": "/neurospin/dico/stounsi/Runs/2025_stounsi_slabeling/models/surfify_cache",
    
    }
    
    model = SingleHemiVGG16BN(input_channels=2, output_dim=512, params=my_params)
    print ("start")
    checkpoint = torch.load(ckpt_path, map_location="cuda",weights_only=False)
    print ("loaded")
    raw_weights = checkpoint["state_dict"]
    
    # D. Nettoyage des clés de l'état dict
    clean_weights = {}
    for key, value in raw_weights.items():
        if key.startswith("backbone."):
            new_key = key.replace("backbone.", "") 
            clean_weights[new_key] = value
            
    missing, unexpected = model.load_state_dict(clean_weights, strict=True)
    
    print(f"✅ Chargé ! (Manquants: {len(missing)}, Inattendus: {len(unexpected)})")
    return model


if __name__ == "__main__":
    ckpt_file = "/neurospin/dico/stounsi/Runs/2025_stounsi_spherical_ssl/outputs/2026-01-29/12-06-12/logs/retest_run21_right_hemisphereico6/version_0/checkpoints/last.ckpt" 
    
    # torch.save({'state_dict': {'backbone.final_proj.weight': torch.randn(512, 32), 'backbone.final_proj.bias': torch.randn(512)}}, ckpt_file)

    try:
        model = load_simple_model(ckpt_file)
        
        print("\n🔍 Inspection des layers :")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f" -> Trouvé un Linear : {name} -> {module}")

        dummy_input = torch.randn(2, 2, 40962)
        
        print(f"\n🧪 Test Forward (Shape: {dummy_input.shape})")
        # model.eval()
        # output = model(dummy_input)
        # print(f"Shape de sortie : {output.shape}")

    except Exception as e:
        print(f"Erreur (C'est normal si tu n'as pas le vrai .ckpt ou surfify ici) : {e}")