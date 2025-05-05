import os
from git import Repo

REPO_URL = "https://github.com/zeeguu/api"
REPO_FOLDER: str = "./Zeeguu-Api"

BLACKLISTED_FOLDERS = {"test"}

VALID_EXTENSIONS = {".py"}

def should_exclude(path):
    parts = path.split(os.sep)
    for part in parts:
        if part in BLACKLISTED_FOLDERS:
            return True
    return False

def count_loc(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
        for file in files:
            full_path = os.path.join(root, file)
            if should_exclude(full_path):
                continue
            if VALID_EXTENSIONS and not file.endswith(tuple(VALID_EXTENSIONS)):
                continue
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    lines = [line for line in f if line.strip()]
                    total_lines += len(lines)
            except Exception as e:
                print(f"Could not read {full_path}: {e}")
    return total_lines

if not os.path.exists(REPO_FOLDER):
    Repo.clone_from(REPO_URL, REPO_FOLDER)
else:
    print(f"{REPO_FOLDER!s} already exists, skipping clone.")

loc = count_loc(os.path.join(REPO_FOLDER, "zeeguu"))
print(f"Total non-empty lines of code (excluding blacklist): {loc}")
