# models/factory.py
import torch
from surfify.utils import number_of_ico_vertices
# Import des modèles
from models.model import MySphericalUNet, Spherical2DCNN, SphericalUNet
from torchinfo import summary

from models.model import SphericalUNetFromPretrained
from models.load_model import SingleHemiVGG16BN
import traceback
from omegaconf import OmegaConf
from models.model import SphericalDinoV2
from models.model import SphericalDinoV3,SphericalDinoV3Linear

def get_model(cfg, device):
    name = cfg.model.name
    

    if name == "PretrainedUNet":
        params=cfg.model.params
        print(f"🏗️ Construction du U-Net à partir de : {params.pretrained_ckpt}")
        
        backbone_conf = OmegaConf.to_container(params.backbone_params, resolve=True)
        
        backbone_conf['input_order'] = cfg.data.ico_order 
        backbone_conf['cachedir'] = cfg.paths.cache_dir
        
        if params.use_3_channels : 
            backbone = SingleHemiVGG16BN(
                input_channels=cfg.data.in_channels-1,
                output_dim=512,
                params=backbone_conf
            )
        else : 
            backbone = SingleHemiVGG16BN(
                input_channels=cfg.data.in_channels,
                output_dim=512,
                params=backbone_conf
            )

        checkpoint = torch.load(params.pretrained_ckpt, map_location=device, weights_only=False)
        raw_weights = checkpoint["state_dict"]
        clean_weights = {}
        
        for key, value in raw_weights.items():
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "")
                clean_weights[new_key] = value
        
        msg = backbone.load_state_dict(clean_weights, strict=False)
        print(f"✅ Poids du backbone chargés. (Missing: {len(msg.missing_keys)})")
        
        model = SphericalUNetFromPretrained(
            backbone=backbone,
            num_classes=cfg.data.num_classes,
            freeze_encoder=params.freeze_encoder,
            use_3_channels=params.use_3_channels,
            dropout=params.dropout,
            use_skip_adapter=params.get('use_skip_adapter', False)
        )
        print('✅ Modèle PretrainedUNet instancié proprement.')

    params = dict(cfg.model.params)
    params['in_channels'] = cfg.data.in_channels
    params['out_channels'] = cfg.data.num_classes
    params['in_order'] = cfg.data.ico_order

    if 'cachedir' not in params:
        params['cachedir'] = cfg.paths.cache_dir

    if name == "SphericalUNet":
        model = SphericalUNet(**params)
    elif name == "Spherical2DCNN":
        model = Spherical2DCNN(in_channels=params['in_channels'], out_channels=params['out_channels'])
    if name == "SphericalDinoV3":
        print("🦖 Instanciation de SphericalDinoV3 (Hugging Face)...")
        model = SphericalDinoV3(
            in_channels=cfg.data.in_channels,
            out_channels=cfg.data.num_classes,
            img_size=cfg.model.params.get('img_size', 528),
            ico_order=cfg.data.ico_order,
            dino_model_name=cfg.model.params.get('dino_version', 'facebook/dinov3-vits16-pretrain-lvd1689m'),
            patch_size=16 
        )
        return model.to(device)
    elif name == "SphericalDinoV3Linear":
        print("🦖 Instanciation de SphericalDinoV3...")
        model = SphericalDinoV3Linear(
            in_channels=cfg.data.in_channels,
            out_channels=cfg.data.num_classes,
            img_size=params.get('img_size', 518),
            ico_order=cfg.data.ico_order,
            dino_model_name=params.get('dino_version', 'facebook/dinov3-vits16-pretrain-lvd1689m'),
            unfreeze_blocks=params.get('unfreeze_blocks', 2)
        )
        print(f"   Modèle créé : Input {cfg.data.in_channels} -> DINO -> {cfg.data.num_classes} classes")
        print (summary(model, input_size=(1, params['in_channels'], number_of_ico_vertices(params['in_order']),)))
        return model.to(device)
    elif name == "SphericalDinoV2":
        print("🦖 Instanciation de SphericalDinoV2...")
        model = SphericalDinoV2(
            in_channels=cfg.data.in_channels,
            out_channels=cfg.data.num_classes,
            img_size=params.get('img_size', 518),
            ico_order=cfg.data.ico_order,
            dino_size=params.get('dino_version', 'dinov2_vits14'),
            unfreeze_blocks=params.get('unfreeze_blocks', 2)
        )
        print(f"   Modèle créé : Input {cfg.data.in_channels} -> DINO -> {cfg.data.num_classes} classes")
        print (summary(model, input_size=(1, params['in_channels'], number_of_ico_vertices(params['in_order']),)))
        return model.to(device)
    elif name == "SPHARM_Net":
        from SPHARMNet.spharmnet.core.models import SPHARM_Net
        
        m_params = cfg.model.params
        
        in_channels = cfg.data.in_channels
        if cfg.model.get("use_single_channel", False):
           
            in_channels = 1
        if params['in_order']== 7 : 
            sp=m_params.get("sphere", f"{cfg.paths.base_dir}/icosphere_order_7.vtk")
        elif params['in_order']== 6 : 
            sp=m_params.get("sphere", f"{cfg.paths.base_dir}/icosphere_order_6.vtk")
        elif params['in_order']== 5 : 
            sp=m_params.get("sphere", f"{cfg.paths.base_dir}/icosphere_order_5.vtk")        
        

        model = SPHARM_Net(
            sphere=sp,
            device=device,
            in_ch=in_channels,
            n_class=cfg.data.num_classes,
            C=m_params.C,
            L=m_params.L,
            D=m_params.D,
            threads=cfg.data.batch_size
        )
    
    print (summary(model, input_size=(1, params['in_channels'], number_of_ico_vertices(params['in_order']),)))
    return model.to(device)