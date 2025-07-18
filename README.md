# Medical-AI-Assistant
 Medical AI Assistant for Fracture Detection using YOLOv8, GPT-2, K-Means, and Logistic Regression. Built with Flask, Transformers, FAISS, and trained on medical X-ray datasets from Kaggle. 
 
Project Title: Medical AI Assistant 


Objective: Implement a machine learning system to detect bone fractures and provide medical insights. 

Key Features:

Image-based fracture detection using YOLOv8. 

Image captioning and visual context extraction with Vision Transformers (e.g., BLIP). 

Semantic search on medical text using Sentence Transformers. 

AI-driven generation of medical insights and contextual information with GPT-2. 





Patient segmentation based on symptom severity using K-Means Clustering. 

Fracture identification prediction (yes/no) based on symptoms using Logistic Regression. 


Models Used: YOLOv8, Vision Transformers (BLIP), Natural Language Processing (NLP) Transformers (Sentence Transformers, GPT-2). 


Datasets: X-ray images of bone fractures from Kaggle (labeled in YOLO format) and a medical text corpus on fractures and treatment. 


System Architecture: Flask Web App, YOLOv8 model, BLIP, Sentence Transformer + FAISS, scikit-learn ML models (Logistic Regression & K-Means). 


Technologies/Libraries: YOLOv8, BLIP, Sentence Transformers (all-MiniLM-L6-v2), GPT-2, Flask, scikit-learn, FAISS, PIL. 





Installation/Setup: (This section would detail how to set up the environment and run the application, e.g., pip install -r requirements.txt, clone repository, etc. - Information not explicitly in sources, but essential for a README)

Usage: (This section would explain how to use the application, e.g., upload an image, enter symptoms, perform searches. - Information not explicitly in sources, but essential for a README)


Future Enhancements: Integration of larger medical datasets, domain-specific transformer fine-tuning, and deployment via Docker + GCP for scalability. 