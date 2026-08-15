import os
import shutil
import random
import glob

def organize_dataset():
    # Base paths
    base_dir = r"F:\UI\SKIN DATASET"
    target_dir = os.path.join(base_dir, "binary_dataset")
    skin_target = os.path.join(target_dir, "skin")
    not_skin_target = os.path.join(target_dir, "not_skin")
    
    # Create target directories
    os.makedirs(skin_target, exist_ok=True)
    os.makedirs(not_skin_target, exist_ok=True)
    
    # 1. Collect skin images
    skin_src_dir = os.path.join(base_dir, "Oily-Dry-Skin-Types")
    skin_images = glob.glob(os.path.join(skin_src_dir, "**", "*.jpg"), recursive=True)
    
    print(f"Found {len(skin_images)} skin images.")
    
    # Copy skin images
    for i, img_path in enumerate(skin_images):
        shutil.copy(img_path, os.path.join(skin_target, f"skin_{i}.jpg"))
        
    num_skin = len(skin_images)
    
    # 2. Collect not_skin images (Intel + Animals)
    not_skin_src_dirs = [
        os.path.join(base_dir, "raw-img"),
        os.path.join(base_dir, "seg_train"),
        os.path.join(base_dir, "seg_test")
    ]
    
    all_not_skin_images = []
    for d in not_skin_src_dirs:
        if os.path.exists(d):
            all_not_skin_images.extend(glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True))
            all_not_skin_images.extend(glob.glob(os.path.join(d, "**", "*.jpeg"), recursive=True))
            
    print(f"Found {len(all_not_skin_images)} total not_skin images.")
    
    # Balance 1:1
    if len(all_not_skin_images) > num_skin:
        random.seed(42)
        selected_not_skin = random.sample(all_not_skin_images, num_skin)
    else:
        selected_not_skin = all_not_skin_images
        
    print(f"Selecting {len(selected_not_skin)} not_skin images to balance the dataset.")
    
    # Copy not_skin images
    for i, img_path in enumerate(selected_not_skin):
        ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(not_skin_target, f"not_skin_{i}{ext}"))
        
    print(f"\nDataset organized successfully at: {target_dir}")
    print(f" - Skin class: {len(os.listdir(skin_target))} images")
    print(f" - Not Skin class: {len(os.listdir(not_skin_target))} images")

if __name__ == "__main__":
    organize_dataset()
