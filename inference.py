# teste_inferencia_completa.py
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

# === Funções de métricas ===
def dice_score_np(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    inter = (y_true * y_pred).sum()
    return (2. * inter + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def iou_score_np(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - inter
    return (inter + smooth) / (union + smooth)

def accuracy_np(y_true, y_pred):
    return (y_true == y_pred).mean()

def specificity_np(y_true, y_pred, smooth=1e-6):
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    return (tn + smooth) / (tn + fp + smooth)

def sensitivity_np(y_true, y_pred, smooth=1e-6):
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    return (tp + smooth) / (tp + fn + smooth)

# === Configurações ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# images_dir = "treino/oralepitheliumdb-aumentado/test/images"
# masks_dir = "treino/oralepitheliumdb-aumentado/test/masks"

images_dir = "dataset/train/test/images"
masks_dir = "dataset/train/test/masks"

threshold = 0.5

# === Carregar modelo TorchScript ===
model_path = "treino/best/fpnnet_resnet_NoAug.pt"
model = torch.jit.load(model_path, map_location=device)
model.eval()
print("Modelo carregado:", model_path)

# === Transformação de entrada ===
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# === Loop sobre todas imagens ===
metrics = {
    "dice": [], "iou": [], "accuracy": [], "specificity": [], "sensitivity": []
}
filenames = []

for fname in sorted(os.listdir(images_dir)):
    if not fname.lower().endswith(".png"):
        continue
    
    # Carregar imagem e máscara
    img_path = os.path.join(images_dir, fname)
    mask_path = os.path.join(masks_dir, fname)  # mesmo nome
    
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize((256,256), resample=Image.NEAREST)
    
    x = transform(image).unsqueeze(0).to(device)
    y_true = np.array(mask) > 0  # binário 0/1

    # Inferência
    with torch.no_grad():
        pred = model(x)
        pred_np = torch.sigmoid(pred)[0,0].cpu().numpy()
        y_pred = (pred_np > threshold).astype(np.uint8)

    # Métricas
    metrics["dice"].append(dice_score_np(y_true, y_pred))
    metrics["iou"].append(iou_score_np(y_true, y_pred))
    metrics["accuracy"].append(accuracy_np(y_true, y_pred))
    metrics["specificity"].append(specificity_np(y_true, y_pred))
    metrics["sensitivity"].append(sensitivity_np(y_true, y_pred))
    filenames.append(fname)

# === Exibir resultados médios ===
print("=== Métricas médias ===")
for k, v in metrics.items():
    print(f"{k}: {np.mean(v):.4f}")

# === Exibir um exemplo ===
example_idx = 0
example_image_path = os.path.join(images_dir, filenames[example_idx])
example_mask_path = os.path.join(masks_dir, filenames[example_idx])

image = Image.open(example_image_path).convert("RGB")
mask = np.array(Image.open(example_mask_path).convert("L").resize((256,256), resample=Image.NEAREST)) > 0

x = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(x)
    pred_mask = (torch.sigmoid(pred)[0,0].cpu().numpy() > threshold).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Imagem original")
plt.imshow(image)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Máscara real")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Predição")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()
