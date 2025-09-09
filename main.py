from models.RetinaLite_Net import RetinaLiteNet
from models.UNeXt import UNeXt
from models.LTMSegNet import LTMSegNet
from models.ESDMR_Net import ESRMNet
from models.MLU_Net import MLUNet
from models.LV_Net import LV_UNet
from models.LEA_Net import LeaNet
from models.LAMFF_Net import LAMFFNet
from models.LWBNA_Net import LWBNA_Unet
from models.PMFS_Net import PMFSNet
from dataset import *
import torch
import torch.optim as optim
import torch.nn as nn
from train import train_model
from utils import *

model_1 = UNeXt(num_classes=1, input_channels=3)
model_2 = LTMSegNet(in_channels=3, base_ch=(16,32,64,128), num_classes=1)
model_3 = ESRMNet(in_channels=3, num_classes=1)
model_4 = RetinaLiteNet(in_channels=3, num_classes=1, num_heads=4, d_k=32)
model_5 = MLUNet(in_channels=3, out_channels=1)
model_6 = LV_UNet()
model_7 = LeaNet(in_channels=3, out_channels=1)
model_8 = LAMFFNet(in_channels=3, num_classes=1, base_c=32)
model_9 = LWBNA_Unet(in_channels=3, num_classes=1)
model_10 = PMFSNet(in_channels=3, n_classes=1)

criterion_UNeXt = BCEDiceLoss()
criterion_LTMSegNet = LTMSegNet_Loss()
criterion_ESRM = ESDMRLoss()
criterion_RetinaLite = Retina_CombinedLoss()
criterion_MLUNet = nn.BCEWithLogitsLoss()
criterion_LVNet = BCEDiceLoss()
criterion_LEANet = BCEDiceLoss()
criterion_LAMFFNet = nn.BCEWithLogitsLoss()
criterion_LWBNANet = LWBNALoss()
criterion_PMFSNet = DiceLoss()


criterions = [
    criterion_UNeXt,
    criterion_LTMSegNet,
    criterion_ESRM,
    criterion_RetinaLite,
    criterion_MLUNet,
    criterion_LVNet,
    criterion_LEANet,
    criterion_LAMFFNet,
    criterion_LWBNANet,
    criterion_PMFSNet
]

models = [
    model_1, 
    model_2, 
    model_3, 
    model_4, 
    model_5, 
    model_6, 
    model_7, 
    model_8, 
    model_9, 
    model_10
]
models_names = [
    "UNeXt",
    "LTMSegNet",
    "ESRMNet",
    "RetinaLiteNet",
    "MLUNet",
    "LV_UNet",
    "LeaNet",
    "LAMFFNet",
    "LWBNA_Unet",
    "PMFSNet"    
]
datsets = [
    "CVC-ClinicDB", 
    "Kvasir-SEG", 
    "Retina-Blood-Vessel", 
    "Breast-Ultrasound-Images", 
    "Skin-Lesion",
    "Brain-Tumor" 
]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

for dataset in datsets:
    if dataset == "CVC-ClinicDB":
        train_loader, test_loader = dataset_cvc(batch_size=4)
    elif dataset == "Kvasir-SEG":
        train_loader, test_loader = dataset_kvasir(batch_size=4)
    elif dataset == "Breast-Ultrasound-Images":
        train_loader, test_loader = dataset_breast_cancer(batch_size=4)
    elif dataset == "Brain-Tumor":
        train_loader, test_loader = dataset_brain_tumor(batch_size=4)
        x, y = next(iter(train_loader))
        print(f"Train batch shape: {x.shape}, {y.shape}")
    elif dataset == "Skin-Lesion":
        train_loader, test_loader = dataset_skin_lesion(batch_size=4)
    elif dataset == "Retina-Blood-Vessel":
        train_loader, test_loader = dataset_retina(batch_size=4)
    else:
        raise ValueError(f"Dataset {dataset} is not supported.")
    for model, model_name, criterion in zip(models, models_names, criterions):
        print(f"Training {model_name} on {dataset} dataset...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, model_name, criterion, train_loader, test_loader, device, dataset)  # Adjust epochs as needed


print ("=="*50)
print("Congratulations 🎉 All models trained succesfully!!!")
print ("=="*50)