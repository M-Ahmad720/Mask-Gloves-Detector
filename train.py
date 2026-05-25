from ultralytics import YOLO
from pathlib import Path
import torch


# CONFIG

BASE_DIR     = Path("datasets")
MASK_YAML    = str((BASE_DIR / "mask_dataset"   / "data.yaml").resolve())
GLOVES_YAML  = str((BASE_DIR / "gloves_dataset" / "data.yaml").resolve())

EPOCHS       = 50
IMG_SIZE     = 640
BATCH_SIZE   = 16   # reduce to 8 if you get memory errors

device = 0 if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Training on: {'GPU ✅' if device == 0 else 'CPU (slower)'}\n")


# TRAIN MASK DETECTOR

def train_mask():
    print("=" * 50)
    print("🎭  Training MASK detector...")
    print("=" * 50)

    model = YOLO("yolov8n.pt")  # loads pretrained weights automatically
    model.train(
        data       = MASK_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = device,
        project    = "runs/mask",
        name       = "mask_detector",
        exist_ok   = True,
        pretrained = True,
        verbose    = True
    )
    print("\n✅ Mask detector training complete!")



# TRAIN GLOVES DETECTOR

def train_gloves():
    print("=" * 50)
    print("🧤  Training GLOVES detector...")
    print("=" * 50)

    model = YOLO("yolov8n.pt")
    model.train(
        data       = GLOVES_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        device     = device,
        project    = "runs/gloves",
        name       = "gloves_detector",
        exist_ok   = True,
        pretrained = True,
        verbose    = True
    )
    print("\n✅ Gloves detector training complete!")



# RUN

if __name__ == "__main__":
    train_mask()
    train_gloves()
    print("\n🎉 Both models trained successfully!")
    print("📁 Mask   model saved → runs/mask/mask_detector/weights/best.pt")
    print("📁 Gloves model saved → runs/gloves/gloves_detector/weights/best.pt")