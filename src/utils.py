# src/utils.py
import os
def get_class_indices_from_dir(train_dir):
    subdirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
    return {name: idx for idx, name in enumerate(subdirs)}
