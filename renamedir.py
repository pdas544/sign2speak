import os

def rename_subdirectories(parent_dir):
    # Get all subdirectories (ignore files)
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    subdirs.sort()  # Optional: sort for consistent renaming

    for idx, old_name in enumerate(subdirs):
        old_path = os.path.join(parent_dir, old_name)
        new_path = os.path.join(parent_dir, str(idx))
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    parent_dir = input("Enter the parent directory path: ")

rename_subdirectories(parent_dir)