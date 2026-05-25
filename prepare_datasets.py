import os
import shutil
import random
from pathlib import Path

# CONFIG — update these paths to match your PC

BASE_DIR        = Path("datasets")
GLOVES_DIR      = BASE_DIR / "gloves_dataset"
MASK_DIR        = BASE_DIR / "mask_dataset"
SPLIT_RATIO     = 0.8   # 80% train, 20% valid


# STEP 1: Split gloves dataset → train / valid

def split_gloves_dataset():
    print("\n Splitting gloves dataset into train/valid...")

    src_images = GLOVES_DIR / "train" / "images"
    src_labels = GLOVES_DIR / "train" / "labels"

    valid_images = GLOVES_DIR / "valid" / "images"
    valid_labels = GLOVES_DIR / "valid" / "labels"
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    all_images = list(src_images.glob("*.jpg")) + \
                 list(src_images.glob("*.jpeg")) + \
                 list(src_images.glob("*.png"))

    random.seed(42)
    random.shuffle(all_images)

    split_idx   = int(len(all_images) * SPLIT_RATIO)
    train_files = all_images[:split_idx]
    valid_files = all_images[split_idx:]

    # Move valid files out of train
    for img_path in valid_files:
        lbl_path = src_labels / (img_path.stem + ".txt")

        shutil.move(str(img_path), str(valid_images / img_path.name))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(valid_labels / lbl_path.name))

    print(f" Train images : {len(train_files)}")
    print(f" Valid  images : {len(valid_files)}")


# STEP 2: Fix data.yaml for GLOVES

def fix_gloves_yaml():
    print("\n Fixing gloves data.yaml...")

    yaml_path = GLOVES_DIR / "data.yaml"
    abs_train = str((GLOVES_DIR / "train" / "images").resolve())
    abs_valid = str((GLOVES_DIR / "valid" / "images").resolve())

    content = f"""train: {abs_train}
val: {abs_valid}

nc: 2
names: ['with_gloves', 'without_gloves']
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f" Gloves yaml saved → {yaml_path}")


# STEP 3: Fix data.yaml for MASK

def fix_mask_yaml():
    print("\n Fixing mask data.yaml...")

    yaml_path = MASK_DIR / "data.yaml"
    abs_train = str((MASK_DIR / "train" / "images").resolve())
    abs_valid = str((MASK_DIR / "valid" / "images").resolve())

    content = f"""train: {abs_train}
val: {abs_valid}

nc: 3
names: ['mask_weared_incorrect', 'with_mask', 'without_mask']
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f" Mask yaml saved → {yaml_path}")



# STEP 4: Verify folder structure

def verify_structure():
    print("\n Verifying dataset structure...")

    checks = [
        GLOVES_DIR / "train" / "images",
        GLOVES_DIR / "train" / "labels",
        GLOVES_DIR / "valid" / "images",
        GLOVES_DIR / "valid" / "labels",
        MASK_DIR   / "train" / "images",
        MASK_DIR   / "train" / "labels",
        MASK_DIR   / "valid" / "images",
        MASK_DIR   / "valid" / "labels",
    ]

    all_good = True
    for path in checks:
        count = len(list(path.glob("*"))) if path.exists() else 0
        status = "✅" if path.exists() and count > 0 else "❌"
        if status == "❌":
            all_good = False
        print(f"   {status} {path}  ({count} files)")

    print()
    if all_good:
        print(" All folders verified! Ready for training.")
    else:
        print(" Some folders are missing or empty. Check paths above.")



# RUN ALL STEPS

if __name__ == "__main__":
    split_gloves_dataset()
    fix_gloves_yaml()
    fix_mask_yaml()
    verify_structure()