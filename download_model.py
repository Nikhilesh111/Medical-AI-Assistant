# from huggingface_hub import hf_hub_download
# import os

# # Define the repository ID and the filename you want to download
# repo_id = "Anurag-Jadhav/Bone-Fracture-YOLOv8m"
# filename = "best.pt" # This is directly in the root of this new repo

# # Define the local directory where the file should be saved
# local_dir = "."

# print(f"Attempting to download '{filename}' from '{repo_id}' to '{local_dir}'...")

# try:
#     cached_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
#     print(f"Successfully downloaded '{filename}' to: {cached_path}")
#     print("\nNow you can update your 'vector_db_app.py' to use 'yolo_model = YOLO(\"best.pt\")'")
# except Exception as e:
#     print(f"Error during download: {e}")
#     print("Please ensure you have an internet connection and the repository/file name is correct.")
#     print("If it's a gated model, make sure you've accepted the terms and logged in with 'huggingface-cli login'")