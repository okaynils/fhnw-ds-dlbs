import os
import json

from sklearn.model_selection import train_test_split

def custom_split_dataset_with_det(
    base_data_path,
    base_labels_path,
    det_train_path,
    det_val_path,
    val_subfolder='val',
    train_subfolder='train',
    test_size=454,
    random_state=1337
):
    with open(det_train_path) as f:
        det_train_data = json.load(f)
    with open(det_val_path) as f:
        det_val_data = json.load(f)

    train_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_train_data}
    val_scene_map = {det["name"]: det["attributes"]["scene"] for det in det_val_data}

    train_images_dir = os.path.join(base_data_path)
    train_labels_dir = os.path.join(base_labels_path)
    
    train_image_filenames = [f for f in os.listdir(train_images_dir) if f in train_scene_map]
    val_image_filenames = [f for f in os.listdir(train_images_dir) if f in val_scene_map]

    train_image_filenames, test_image_filenames = train_test_split(
        train_image_filenames, test_size=test_size, random_state=random_state
    )
    
    print()
    print("--- Split Sizes ---")
    print(f"- Train Images: {len(train_image_filenames)}")
    print(f"- Val Images: {len(val_image_filenames)}")
    print(f"- Test Images: {len(test_image_filenames)}")
    print()
    
    return {
        'train': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': train_image_filenames,
            'scene_map': {k: train_scene_map[k] for k in train_image_filenames},
        },
        'val': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': val_image_filenames,
            'scene_map': {k: val_scene_map[k] for k in val_image_filenames},
        },
        'test': {
            'data_folder': train_images_dir,
            'labels_folder': train_labels_dir,
            'image_filenames': test_image_filenames,
            'scene_map': {k: train_scene_map[k] for k in test_image_filenames},
        }
    }
    
def check_dataset_overlap(train_filenames, val_filenames, test_filenames):
    train_set = set(train_filenames)
    val_set = set(val_filenames)
    test_set = set(test_filenames)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    print("--- Overlap Report ---")
    if train_val_overlap:
        print(f"🚨 Overlap detected between train and validation sets! {len(train_val_overlap)} samples.")
        print(f"Overlapping samples: {list(train_val_overlap)}")
    else:
        print("✔️ No overlap detected between train and validation sets.")

    if train_test_overlap:
        print(f"🚨 Overlap detected between train and test sets! {len(train_test_overlap)} samples.")
        print(f"Overlapping samples: {list(train_test_overlap)}")
    else:
        print("✔️ No overlap detected between train and test sets.")

    if val_test_overlap:
        print(f"🚨 Overlap detected between validation and test sets! {len(val_test_overlap)} samples.")
        print(f"Overlapping samples: {list(val_test_overlap)}")
    else:
        print("✔️ No overlap detected between validation and test sets.")
    print()
