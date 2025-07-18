# train_yolov8.py

from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Training will run on CPU, which will be significantly slower.")

# 1. Load a pre-trained YOLOv8 model
# You can choose yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium),
# yolov8l.pt (large), yolov8x.pt (extra-large) based on your GPU capabilities.
# 'yolov8m.pt' (medium) is a good balance for many cases.
model = YOLO("yolov8m.pt") # This will download the weights if not already present

# 2. Train the model with your custom dataset
# - 'data': path to your data.yaml file
# - 'epochs': number of training iterations. Start with 50-100 and increase if needed.
#             Training takes a long time, so start small to confirm everything works.
# - 'imgsz': input image size. 640 is standard for YOLOv8.
# - 'batch': number of images per batch. Adjust based on your GPU memory.
#            If you get "out of memory" errors, reduce this number (e.g., to 8, 4, or 2).
# - 'name': A name for your training run. Results will be saved in 'runs/detect/<name>'
# - 'project': Top-level directory for your runs (default is 'runs')
results = model.train(
    data="data.yaml",   # Path to your data.yaml file
    epochs=100,         # Number of epochs
    imgsz=640,          # Image size
    batch=16,           # Batch size (adjust based on GPU memory)
    name="fracture_detection_run" # Name of this training session
)

# 3. (Optional) Evaluate the trained model on the validation set
# This will calculate metrics like mAP, Precision, Recall.
metrics = model.val()

print("\nTraining completed!")
print(f"Model weights saved to: {model.trainer.save_dir}/weights/best.pt")
print("You can now use this 'best.pt' file in your Flask app for fracture detection.")