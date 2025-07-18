
import os
import faiss
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from sentence_transformers import SentenceTransformer
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scaling
from ultralytics import YOLO
import cv2 # Imported but not used in the provided snippet
import re

# --- Configuration and Global Variables ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB max upload size
app.config['STATIC_FOLDER'] = 'static' # For serving processed images

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

print("Loading Sentence Transformer model...")
text_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Sentence Transformer loaded.")

print("Loading Text Generation model (GPT2)...")
text_generator = pipeline("text-generation", model="gpt2")
print("Text Generation model loaded.")

print("Loading Image Captioning model (BLIP)...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_model.to(device)
    print(f"BLIP model loaded on {device}.")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    blip_processor = None
    blip_model = None

print("Loading YOLOv8 Fracture Detection model...")
try:
    yolo_model = YOLO("yolov8m.pt")
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    yolo_model = None

# --- EXPANDED Medical Corpus (as before) ---
initial_medical_corpus = [
    "Headaches are a common ailment often relieved by rest or over-the-counter pain relievers.",
    # "A fractured bone requires immediate medical attention and immobilization.",
    # "Common cold symptoms include runny nose, sore throat, and coughing.",
    # "Diabetes management involves diet, exercise, and medication to control blood sugar levels.",
    # "Heart disease prevention focuses on a healthy lifestyle and regular check-ups.",
    # "Pediatric fevers in infants should be monitored closely and a doctor consulted.",
    # "Chronic back pain can be managed with physical therapy and pain management strategies.",
    # "Sprains and strains often heal with R.I.C.E. (Rest, Ice, Compression, Elevation).",
    # "Asthma is a chronic respiratory condition that requires ongoing treatment.",
    # "Hypertension, or high blood pressure, often has no symptoms but increases stroke risk.",
    # "An allergic reaction can range from mild skin irritation to severe anaphylaxis.",
    # "Early diagnosis of cancer significantly improves treatment outcomes.",
    # "Vaccinations are crucial for preventing infectious diseases.",
    # "Depression is a serious mood disorder that can be treated with therapy and medication.",
    # "Arthritis causes joint inflammation and pain, common in older adults."
]

print("Building initial FAISS index...")
corpus_embeddings = text_embedding_model.encode(initial_medical_corpus)
faiss.normalize_L2(corpus_embeddings)
dimension = corpus_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(corpus_embeddings)
print("FAISS index built.")

# --- K-Means Clustering Setup with StandardScaler ---
print("Initializing K-Means model...")
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init=10 for better stability

# Define StandardScaler
scaler = StandardScaler()

# Dummy data for K-Means (patient features: [age, temp, pain_level, fatigue_score])
# These data points are designed to naturally fall into distinct groups
dummy_patient_data = np.array([
    # Cluster 1: Acute/Severe (higher pain, higher temp, general adult)
    [30, 101.5, 9, 8], [25, 102.0, 8, 9], [40, 100.8, 7, 7],
    
    # Cluster 2: Chronic/Mild (moderate pain, normal temp, older adult)
    [60, 98.0, 4, 5], [65, 97.8, 3, 4], [55, 98.2, 5, 6],
    
    # Cluster 3: Pediatric/Severe (low age, high temp, high pain)
    [10, 102.5, 10, 9], [8, 101.0, 8, 8], [12, 100.0, 7, 7]
])

# Fit the scaler on the dummy data and transform it
scaled_dummy_patient_data = scaler.fit_transform(dummy_patient_data)
kmeans_model.fit(scaled_dummy_patient_data) # Fit K-Means on scaled data
print("K-Means model initialized and fitted with scaled dummy data.")

# --- FIX START: Dynamic Cluster ID to Description Mapping ---
# 1. Get the centroids in the original (unscaled) feature space
original_cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)

# 2. Determine meaningful descriptions for each cluster ID based on centroid characteristics
#    This mapping needs to be determined *after* KMeans has run.
#    We will iterate through the centroids and assign descriptions based on their values.
#    The indices of the features in dummy_patient_data are:
#    [0:age, 1:temp, 2:pain_level, 3:fatigue_score]

cluster_id_to_description_map = {}

for i, center in enumerate(original_cluster_centers):
    age, temp, pain_level, fatigue_score = center

    # Define conditions to identify each cluster type.
    # These conditions are based on the *expected* characteristics of your clusters
    # in the original feature space. Adjust these thresholds as needed based on
    # your actual `original_cluster_centers` values when you inspect them.
    
    # Example logic:
    if age <= 15 and (pain_level >= 7 or temp >= 101.0): # Pediatric and severe symptoms
        cluster_id_to_description_map[i] = "This patient profile aligns with a pediatric/severe symptom cluster."
    elif pain_level >= 7 and temp >= 100.0 and age > 15: # Acute/Severe, but not pediatric
        cluster_id_to_description_map[i] = "This patient profile aligns with an acute/severe symptom cluster."
    else: # The remaining cluster, likely chronic/mild
        cluster_id_to_description_map[i] = "This patient profile aligns with a chronic/mild symptom cluster."

print("K-Means cluster descriptions dynamically mapped based on centroid characteristics.")
print("Original Cluster Centers (inverse transformed):\n", original_cluster_centers)
print("Mapped Descriptions:", cluster_id_to_description_map)
# --- FIX END ---


# --- Helper Functions ---
def generate_sentences(query, num_sentences=3):
    generated = text_generator(query, max_length=70, num_return_sequences=num_sentences,
                               do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
    sentences = [text["generated_text"].strip() for text in generated]
    # Filter out very short or nonsensical generated sentences
    sentences = [s for s in sentences if len(s.split()) > 5 and not s.endswith("...")]
    return sentences

# Function to draw bounding boxes on an image
def draw_boxes_on_image(image_path, detections, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'].values())
        label = det['name']
        confidence = det['confidence']

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Calculate text bounding box to position background rectangle correctly
        text_bbox = draw.textbbox((x1, y1), f"{label} {confidence:.2f}", font=font)
        # Draw a filled rectangle behind the text for better readability
        draw.rectangle([x1, y1 - (text_bbox[3] - text_bbox[1]) - 5, text_bbox[2] + 5, y1], fill="red")

        draw.text((x1 + 2, y1 - (text_bbox[3] - text_bbox[1]) - 5), f"{label} {confidence:.2f}", fill="white", font=font)
    
    img.save(output_path)
    return output_path

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def search_page():
    results = []
    generated_text_results = []
    image_analysis_results = None
    kmeans_cluster_result = None
    processed_image_url = None

    if request.method == "POST":
        user_query = request.form.get("query", "")
        file = request.files.get("image_file")
        kmeans_input_str = request.form.get("kmeans_input", "")
        
        combined_query = user_query # Initialize combined_query

        # 1. Image Analysis (Captioning & Fracture Detection)
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform BLIP captioning
            if blip_processor and blip_model:
                try:
                    raw_image = Image.open(filepath).convert("RGB")
                    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
                    out = blip_model.generate(**inputs)
                    caption_text = blip_processor.decode(out[0], skip_special_tokens=True)
                    image_analysis_results = f"Image Caption: {caption_text}"
                    combined_query += f" {caption_text}" # Add caption to combined query
                except Exception as e:
                    image_analysis_results = f"Error during BLIP captioning: {e}"
            else:
                image_analysis_results = "BLIP image captioning model not loaded."

            # Perform YOLOv8 fracture detection
            if yolo_model:
                try:
                    # YOLO expects image path or PIL Image
                    yolo_results = yolo_model(filepath)
                    detections = []
                    fracture_info_for_llm = []

                    for r in yolo_results:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        class_ids = r.boxes.cls.cpu().numpy()
                        names = r.names

                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[i]
                            conf = scores[i]
                            cls_id = int(class_ids[i])
                            label = names[cls_id]

                            # Assuming 'fracture' is a class detected by yolov8m.pt
                            # Note: General YOLOv8 models (like yolov8m.pt) are not trained on medical images.
                            # For actual fracture detection, you would need a model trained on medical X-rays.
                            # This 'if label == 'fracture'' might not trigger as expected with yolov8m.pt.
                            # For demonstration, we'll just check for any detection above threshold.
                            # If you have a custom trained model for 'fracture', this condition is perfect.
                            if conf > 0.5: # Consider all detections for now, or specifically check if 'fracture' is a class name.
                                detections.append({
                                    'box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                                    'name': label, # Using the detected label
                                    'confidence': conf
                                })
                                fracture_info_for_llm.append(f"a {label} detected at coordinates ({int(x1)},{int(y1)}) to ({int(x2)},{int(y2)}) with confidence {conf:.2f}")

                    if detections:
                        processed_filename = "processed_" + filename
                        processed_filepath = os.path.join(app.config['STATIC_FOLDER'], processed_filename)
                        draw_boxes_on_image(filepath, detections, processed_filepath)
                        processed_image_url = url_for('static_file', filename=processed_filename)

                        fracture_summary = "Detected objects: " + ", ".join(fracture_info_for_llm)
                        image_analysis_results += f"<br><strong>{fracture_summary}</strong>"
                        combined_query += f" {fracture_summary}" # Add fracture info to combined query

                except Exception as e:
                    image_analysis_results = (image_analysis_results or "") + f"<br>Error during YOLOv8 detection: {e}"
            else:
                image_analysis_results = (image_analysis_results or "") + "<br>YOLOv8 model not loaded."

            # Clean up the uploaded file after processing
            os.remove(filepath)

        # 2. Text Search (using vector DB on static corpus)
        if combined_query.strip():
            query_vector = text_embedding_model.encode([combined_query])
            faiss.normalize_L2(query_vector)

            D, I = faiss_index.search(query_vector, 10) # Search for top 10 relevant items
            threshold = 0.65

            unique_results = {}
            for i, score in zip(I[0], D[0]):
                sentence = initial_medical_corpus[i]
                if score >= threshold and sentence not in unique_results:
                    unique_results[sentence] = float(score)

            results = sorted(unique_results.items(), key=lambda item: item[1], reverse=True)


        # 3. Text Generation (using GPT-2) - now influenced by image analysis
        if combined_query.strip():
            if image_analysis_results:
                # Remove HTML tags from image_analysis_results for a cleaner prompt
                clean_image_analysis = re.sub(r'<[^>]+>', '', image_analysis_results)
                prompt = f"Based on the image analysis: '{clean_image_analysis}'. And the user's query: '{user_query}'. Describe the medical implications, potential recovery, or general tips related to the findings."
            else:
                prompt = user_query

            generated_sentences = generate_sentences(prompt)
            if generated_sentences:
                generated_text_results = [(s, "AI Generated Insight") for s in generated_sentences]
            else:
                generated_text_results.append(("Could not generate relevant AI insights.", "AI Generated Insight"))

        # 4. K-Means Clustering (Example for symptom clustering/patient grouping)
        if kmeans_input_str:
            try:
                # Ensure input format matches expected features: [age, temp, pain_level, fatigue_score]
                kmeans_input = np.array([float(x.strip()) for x in kmeans_input_str.split(',')]).reshape(1, -1)

                if kmeans_input.shape[1] == dummy_patient_data.shape[1]:
                    # Scale the new input using the SAME scaler fitted on dummy_patient_data
                    scaled_kmeans_input = scaler.transform(kmeans_input)
                    cluster_id = kmeans_model.predict(scaled_kmeans_input)[0] # Predict on scaled input
                    
                    # --- FIX: Use the dynamically determined mapping ---
                    kmeans_cluster_result = cluster_id_to_description_map.get(cluster_id, "Unknown cluster type.")
                    
                else:
                    kmeans_cluster_result = f"K-Means input requires {dummy_patient_data.shape[1]} comma-separated numbers (e.g., age, temp, pain_level, fatigue_score)."
            except ValueError:
                kmeans_cluster_result = "Invalid K-Means input. Please enter comma-separated numbers."
            except Exception as e:
                kmeans_cluster_result = f"Error processing K-Means input: {e}"

    return render_template("index.html",
                           results=results,
                           generated_text_results=generated_text_results,
                           image_analysis_results=image_analysis_results,
                           kmeans_cluster_result=kmeans_cluster_result,
                           processed_image_url=processed_image_url)

@app.route('/static/<path:filename>')
def static_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')



