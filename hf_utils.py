import os
from huggingface_hub import HfApi, login

def upload_to_hf(repo_name, local_path, token=None, username="ahmed275"):
    api = HfApi()
    if token:
        login(token=token)
    
    full_repo_id = f"{username}/{repo_name}"
    api.create_repo(repo_id=full_repo_id, exist_ok=True)

    # Check if the path is a file or a directory
    if os.path.isfile(local_path):
        print(f"📄 Detected FILE. Uploading: {local_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path), # Keeps the filename
            repo_id=full_repo_id
        )
    elif os.path.isdir(local_path):
        print(f"📂 Detected FOLDER. Uploading contents of: {local_path}")
        api.upload_folder(
            folder_path=local_path,
            repo_id=full_repo_id,
            ignore_patterns=["**/venv/*", "**/__pycache__/*"]
        )
    else:
        print(f"❌ Error: {local_path} is not a valid file or folder.")

# Try running it again with your current path:
# upload_to_hf("sam3-medical-segmentation", "/home/ahma/Medical_Segmentation/pipeline/SAM3/sam3_best.pth", token="your_token")