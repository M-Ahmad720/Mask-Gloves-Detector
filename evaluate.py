from ultralytics import YOLO
from pathlib import Path


# CONFIG

BASE_DIR     = Path("datasets")
MASK_YAML    = str((BASE_DIR / "mask_dataset"   / "data.yaml").resolve())
GLOVES_YAML  = str((BASE_DIR / "gloves_dataset" / "data.yaml").resolve())

MASK_MODEL   = "models/mask_best.pt"
GLOVES_MODEL = "models/gloves_best.pt"


# EVALUATE MASK MODEL

def evaluate_mask():
    print("\n" + "=" * 50)
    print("Evaluating MASK detector...")
    print("=" * 50)

    model   = YOLO(MASK_MODEL)
    metrics = model.val(
        data     = MASK_YAML,
        imgsz    = 640,
        device   = "cpu",
        project  = "runs/evaluation",
        name     = "mask_eval",
        exist_ok = True
    )

    print("\n MASK MODEL RESULTS:")
    print(f"   mAP50        : {metrics.box.map50:.4f}")
    print(f"   mAP50-95     : {metrics.box.map:.4f}")
    print(f"   Precision    : {metrics.box.mp:.4f}")
    print(f"   Recall       : {metrics.box.mr:.4f}")



# EVALUATE GLOVES MODEL

def evaluate_gloves():
    print("\n" + "=" * 50)
    print(" Evaluating GLOVES detector...")
    print("=" * 50)

    model   = YOLO(GLOVES_MODEL)
    metrics = model.val(
        data     = GLOVES_YAML,
        imgsz    = 640,
        device   = "cpu",
        project  = "runs/evaluation",
        name     = "gloves_eval",
        exist_ok = True
    )

    print("\nGLOVES MODEL RESULTS:")
    print(f"   mAP50        : {metrics.box.map50:.4f}")
    print(f"   mAP50-95     : {metrics.box.map:.4f}")
    print(f"   Precision    : {metrics.box.mp:.4f}")
    print(f"   Recall       : {metrics.box.mr:.4f}")


# GRADE THE MODEL

def grade(map50):
    if map50 >= 0.90:
        return " Excellent"
    elif map50 >= 0.75:
        return " Good"
    elif map50 >= 0.60:
        return " Fair - consider more training"
    else:
        return " Poor - needs improvement"



# RUN

if __name__ == "__main__":
    evaluate_mask()
    evaluate_gloves()
    print("\n" + "=" * 50)
    print(" Evaluation Complete!")
    print(" Confusion matrices saved → runs/evaluation/")
    print("=" * 50)