import tarfile
import os

archive_file = "archive.tar.gz"
extract_to_dir = "pose_recognition_model"
os.makedirs(extract_to_dir, exist_ok=True)
with tarfile.open(archive_file, "r:gz") as tar:
    tar.extractall(path=extract_to_dir)

