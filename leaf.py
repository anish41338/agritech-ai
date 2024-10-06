# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
import cv2  # Ensure cv2 is imported
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import numpy as np
import json
import mimetypes
from skimage.metrics import structural_similarity as compare_ssim
from collections import Counter  # For counting class occurrences

# Load the model
filepath = 'plant_disease_prediction_model.h5'
model = load_model(filepath)
print("Model Loaded Successfully")

# Class indices mapping
class_indices = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Treatment solutions
treatment_solutions = {
    'Apple___Apple_scab': "Remove affected leaves and apply fungicides.",
    'Apple___Black_rot': "Prune affected areas and apply appropriate fungicides.",
    'Apple___Cedar_apple_rust': "Use resistant varieties and remove cedar trees nearby.",
    'Blueberry___healthy': "No action required; continue regular care.",
    'Cherry_(including_sour)___Powdery_mildew': "Use fungicides and ensure proper air circulation.",
    'Cherry_(including_sour)___healthy': "No action required; continue regular care.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops and use resistant varieties.",
    'Corn_(maize)___Common_rust_': "Apply fungicides and remove crop residues.",
    'Corn_(maize)___Northern_Leaf_Blight': "Practice crop rotation and apply fungicides.",
    'Corn_(maize)___healthy': "No action required; continue regular care.",
    'Grape___Black_rot': "Remove infected leaves and apply sulfur or other fungicides.",
    'Grape___Esca_(Black_Measles)': "Prune affected vines and improve air circulation.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use fungicides and remove affected leaves.",
    'Grape___healthy': "No action required; continue regular care.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove affected trees and use resistant varieties.",
    'Peach___Bacterial_spot': "Remove infected parts and apply bactericides.",
    'Peach___healthy': "No action required; continue regular care.",
    'Pepper,_bell___Bacterial_spot': "Remove infected plants and use resistant varieties.",
    'Pepper,_bell___healthy': "No action required; continue regular care.",
    'Potato___Early_blight': "Rotate crops and apply fungicides.",
    'Potato___Late_blight': "Use resistant varieties and apply appropriate fungicides.",
    'Potato___healthy': "No action required; continue regular care.",
    'Raspberry___healthy': "No action required; continue regular care.",
    'Soybean___healthy': "No action required; continue regular care.",
    'Squash___Powdery_mildew': "Use fungicides and improve air circulation.",
    'Strawberry___Leaf_scorch': "Water plants properly and remove affected leaves.",
    'Strawberry___healthy': "No action required; continue regular care.",
    'Tomato___Bacterial_spot': "Remove infected plants and use resistant varieties.",
    'Tomato___Early_blight': "Use crop rotation and apply fungicides.",
    'Tomato___Late_blight': "Use resistant varieties and fungicides.",
    'Tomato___Leaf_Mold': "Ensure proper ventilation and use fungicides.",
    'Tomato___Septoria_leaf_spot': "Apply fungicides and remove affected leaves.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use miticides and promote beneficial insects.",
    'Tomato___Target_Spot': "Use fungicides and remove infected leaves.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants; no cure available.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants; no cure available.",
    'Tomato___healthy': "No action required; continue regular care."
}

# Causes of diseases
disease_causes = {
    'Apple___Apple_scab': "Cause: Fungal infection caused by *Venturia inaequalis*.",
    'Apple___Black_rot': "Cause: Fungal infection caused by *Alternaria alternata*.",
    'Apple___Cedar_apple_rust': "Cause: Fungal infection caused by *Gymnosporangium juniperi-virginianae*.",
    'Blueberry___healthy': "Cause: No disease, the plant is healthy.",
    'Cherry_(including_sour)___Powdery_mildew': "Cause: Fungal infection caused by *Podosphaera clandestina*.",
    'Cherry_(including_sour)___healthy': "Cause: No disease, the plant is healthy.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Cause: Fungal infection caused by *Cercospora zeae-maydis*.",
    'Corn_(maize)___Common_rust_': "Cause: Fungal infection caused by *Puccinia sorghi*.",
    'Corn_(maize)___Northern_Leaf_Blight': "Cause: Fungal infection caused by *Exserohilum turcicum*.",
    'Corn_(maize)___healthy': "Cause: No disease, the plant is healthy.",
    'Grape___Black_rot': "Cause: Fungal infection caused by *Guignardia bidwellii*.",
    'Grape___Esca_(Black_Measles)': "Cause: Fungal infection caused by *Phaeomoniella chlamydospora*.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Cause: Fungal infection caused by *Isariopsis clavispora*.",
    'Grape___healthy': "Cause: No disease, the plant is healthy.",
    'Orange___Haunglongbing_(Citrus_greening)': "Cause: Bacterial infection caused by *Candidatus Liberibacter asiaticus*.",
    'Peach___Bacterial_spot': "Cause: Bacterial infection caused by *Xanthomonas arboricola*.",
    'Peach___healthy': "Cause: No disease, the plant is healthy.",
    'Pepper,_bell___Bacterial_spot': "Cause: Bacterial infection caused by *Xanthomonas campestris*.",
    'Pepper,_bell___healthy': "Cause: No disease, the plant is healthy.",
    'Potato___Early_blight': "Cause: Fungal infection caused by *Alternaria solani*.",
    'Potato___Late_blight': "Cause: Fungal infection caused by *Phytophthora infestans*.",
    'Potato___healthy': "Cause: No disease, the plant is healthy.",
    'Raspberry___healthy': "Cause: No disease, the plant is healthy.",
    'Soybean___healthy': "Cause: No disease, the plant is healthy.",
    'Squash___Powdery_mildew': "Cause: Fungal infection caused by *Podosphaera xanthii*.",
    'Strawberry___Leaf_scorch': "Cause: Excessive heat, drought, or pest damage.",
    'Strawberry___healthy': "Cause: No disease, the plant is healthy.",
    'Tomato___Bacterial_spot': "Cause: Bacterial infection caused by *Xanthomonas perforans*.",
    'Tomato___Early_blight': "Cause: Fungal infection caused by *Alternaria solani*.",
    'Tomato___Late_blight': "Cause: Fungal infection caused by *Phytophthora infestans*.",
    'Tomato___Leaf_Mold': "Cause: Fungal infection caused by *Cladosporium fulvum*.",
    'Tomato___Septoria_leaf_spot': "Cause: Fungal infection caused by *Septoria lycopersici*.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Cause: Mite infestation caused by *Tetranychus urticae*.",
    'Tomato___Target_Spot': "Cause: Fungal infection caused by *Corynespora cassiicola*.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Cause: Viral infection caused by *Tomato yellow leaf curl virus (TYLCV)*.",
    'Tomato___Tomato_mosaic_virus': "Cause: Viral infection caused by *Tomato mosaic virus (ToMV)*.",
    'Tomato___healthy': "Cause: No disease, the plant is healthy."
}

crop_damage_data = {
  "2023": {
    "January": {
      "Week 1": {"Healthy": 100, "Diseased": 50},
      "Week 2": {"Healthy": 120, "Diseased": 60},
      "Week 3": {"Healthy": 90, "Diseased": 70},
      "Week 4": {"Healthy": 110, "Diseased": 55}
    },
    "February": {
      "Week 1": {"Healthy": 95, "Diseased": 40},
      "Week 2": {"Healthy": 102, "Diseased": 65},
      "Week 3": {"Healthy": 98, "Diseased": 50},
      "Week 4": {"Healthy": 115, "Diseased": 80}
    },
    "March": {
      "Week 1": {"Healthy": 100, "Diseased": 30},
      "Week 2": {"Healthy": 130, "Diseased": 55},
      "Week 3": {"Healthy": 110, "Diseased": 45},
      "Week 4": {"Healthy": 140, "Diseased": 75}
    },
    "April": {
      "Week 1": {"Healthy": 125, "Diseased": 40},
      "Week 2": {"Healthy": 115, "Diseased": 50},
      "Week 3": {"Healthy": 135, "Diseased": 60},
      "Week 4": {"Healthy": 105, "Diseased": 45}
    },
    "May": {
      "Week 1": {"Healthy": 110, "Diseased": 35},
      "Week 2": {"Healthy": 130, "Diseased": 55},
      "Week 3": {"Healthy": 120, "Diseased": 40},
      "Week 4": {"Healthy": 135, "Diseased": 65}
    },
    "June": {
      "Week 1": {"Healthy": 150, "Diseased": 70},
      "Week 2": {"Healthy": 140, "Diseased": 60},
      "Week 3": {"Healthy": 135, "Diseased": 50},
      "Week 4": {"Healthy": 145, "Diseased": 80}
    },
    "July": {
      "Week 1": {"Healthy": 130, "Diseased": 65},
      "Week 2": {"Healthy": 120, "Diseased": 55},
      "Week 3": {"Healthy": 135, "Diseased": 60},
      "Week 4": {"Healthy": 140, "Diseased": 70}
    },
    "August": {
      "Week 1": {"Healthy": 140, "Diseased": 60},
      "Week 2": {"Healthy": 135, "Diseased": 55},
      "Week 3": {"Healthy": 150, "Diseased": 65},
      "Week 4": {"Healthy": 160, "Diseased": 80}
    },
    "September": {
      "Week 1": {"Healthy": 135, "Diseased": 50},
      "Week 2": {"Healthy": 145, "Diseased": 60},
      "Week 3": {"Healthy": 155, "Diseased": 70},
      "Week 4": {"Healthy": 150, "Diseased": 80}
    },
    "October": {
      "Week 1": {"Healthy": 120, "Diseased": 40},
      "Week 2": {"Healthy": 125, "Diseased": 55},
      "Week 3": {"Healthy": 130, "Diseased": 65},
      "Week 4": {"Healthy": 140, "Diseased": 75}
    },
    "November": {
      "Week 1": {"Healthy": 110, "Diseased": 35},
      "Week 2": {"Healthy": 120, "Diseased": 45},
      "Week 3": {"Healthy": 125, "Diseased": 50},
      "Week 4": {"Healthy": 130, "Diseased": 60}
    },
    "December": {
      "Week 1": {"Healthy": 115, "Diseased": 40},
      "Week 2": {"Healthy": 125, "Diseased": 50},
      "Week 3": {"Healthy": 130, "Diseased": 55},
      "Week 4": {"Healthy": 135, "Diseased": 65}
    }
  },
  "2022": {
    "January": {
      "Week 1": {"Healthy": 110, "Diseased": 60},
      "Week 2": {"Healthy": 115, "Diseased": 55},
      "Week 3": {"Healthy": 120, "Diseased": 50},
      "Week 4": {"Healthy": 130, "Diseased": 70}
    },
    "February": {
      "Week 1": {"Healthy": 90, "Diseased": 40},
      "Week 2": {"Healthy": 95, "Diseased": 45},
      "Week 3": {"Healthy": 100, "Diseased": 50},
      "Week 4": {"Healthy": 110, "Diseased": 60}
    },
    "March": {
      "Week 1": {"Healthy": 120, "Diseased": 65},
      "Week 2": {"Healthy": 130, "Diseased": 60},
      "Week 3": {"Healthy": 125, "Diseased": 55},
      "Week 4": {"Healthy": 140, "Diseased": 75}
    },
    "April": {
      "Week 1": {"Healthy": 130, "Diseased": 55},
      "Week 2": {"Healthy": 135, "Diseased": 60},
      "Week 3": {"Healthy": 140, "Diseased": 70},
      "Week 4": {"Healthy": 145, "Diseased": 80}
    },
    "May": {
      "Week 1": {"Healthy": 120, "Diseased": 50},
      "Week 2": {"Healthy": 130, "Diseased": 60},
      "Week 3": {"Healthy": 125, "Diseased": 55},
      "Week 4": {"Healthy": 140, "Diseased": 75}
    },
    "June": {
      "Week 1": {"Healthy": 150, "Diseased": 80},
      "Week 2": {"Healthy": 140, "Diseased": 75},
      "Week 3": {"Healthy": 135, "Diseased": 65},
      "Week 4": {"Healthy": 155, "Diseased": 90}
    },
    "July": {
      "Week 1": {"Healthy": 130, "Diseased": 55},
      "Week 2": {"Healthy": 135, "Diseased": 60},
      "Week 3": {"Healthy": 140, "Diseased": 65},
      "Week 4": {"Healthy": 150, "Diseased": 75}
    },
    "August": {
      "Week 1": {"Healthy": 145, "Diseased": 60},
      "Week 2": {"Healthy": 135, "Diseased": 55},
      "Week 3": {"Healthy": 150, "Diseased": 70},
      "Week 4": {"Healthy": 160, "Diseased": 80}
    },
    "September": {
      "Week 1": {"Healthy": 140, "Diseased": 55},
      "Week 2": {"Healthy": 145, "Diseased": 60},
      "Week 3": {"Healthy": 150, "Diseased": 65},
      "Week 4": {"Healthy": 160, "Diseased": 75}
    },
    "October": {
      "Week 1": {"Healthy": 125, "Diseased": 50},
      "Week 2": {"Healthy": 135, "Diseased": 60},
      "Week 3": {"Healthy": 130, "Diseased": 55},
      "Week 4": {"Healthy": 145, "Diseased": 75}
    },
    "November": {
      "Week 1": {"Healthy": 120, "Diseased": 40},
      "Week 2": {"Healthy": 130, "Diseased": 50},
      "Week 3": {"Healthy": 125, "Diseased": 45},
      "Week 4": {"Healthy": 140, "Diseased": 60}
    },
    "December": {
      "Week 1": {"Healthy": 110, "Diseased": 35},
      "Week 2": {"Healthy": 115, "Diseased": 45},
      "Week 3": {"Healthy": 120, "Diseased": 50},
      "Week 4": {"Healthy": 125, "Diseased": 60}
    }
  }
}

def predict_image_class(model, image, class_indices):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    print(f"Predictions: {preds}")  # Debugging line
    predicted_class = np.argmax(preds, axis=1)[0]
    print(f"Predicted Class Index: {predicted_class}")  # Debugging line
    return class_indices[str(predicted_class)], preds[0][predicted_class]


# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Load class indices (replace 'class_indices.json' with your class labels)
with open('static/class_indices.json') as f:
    class_indices = json.load(f)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to compute the structural similarity index (SSIM) between two images
def is_different(imageA, imageB, threshold=0.5):
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = compare_ssim(grayA, grayB, full=True)
    return score < threshold  # If SSIM score is below the threshold, images are different

# Function to extract unique frames from video based on SSIM
def extract_unique_frames(video_path, frame_rate=1, ssim_threshold=0.5):
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # Get frames per second

    frames_dir = 'frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_id = 0
    success, current_frame = video_capture.read()
    last_saved_frame = None

    while success:
        if frame_id % (fps * frame_rate) == 0:  # Extract every `frame_rate` seconds
            if last_saved_frame is None or is_different(current_frame, last_saved_frame, threshold=ssim_threshold):
                # Save the current frame only if it's different from the last saved frame
                frame_filename = os.path.join(frames_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_filename, current_frame)
                last_saved_frame = current_frame
        frame_id += 1
        success, current_frame = video_capture.read()

    video_capture.release()
    return frames_dir

# Function to preprocess image and make prediction
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_frame(frame_path):
    image = load_and_preprocess_image(frame_path)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Route to serve the frontend page
@app.route('/analyze')
def index():
    return render_template('index2.html')

@app.route('/frames/<path:filename>')
def serve_frame(filename):
    return send_from_directory('/frames', filename)


# API to handle file upload (both image and video)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Determine the file type
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type and mime_type.startswith('image'):
        # Handle image file
        predicted_class = predict_frame(file_path)
        return jsonify({'file': file.filename, 'prediction': predicted_class})

    elif mime_type and mime_type.startswith('video'):
        # Handle video file
        frames_dir = extract_unique_frames(file_path)
        results = []
        predictions = []  # To store predicted class names for all frames

        for frame_filename in os.listdir(frames_dir):
            frame_path = os.path.join(frames_dir, frame_filename)
            predicted_class = predict_frame(frame_path)
            predictions.append(predicted_class)  # Collect predictions
            results.append({
                'frame': frame_filename,
                'prediction': predicted_class
            })

        # Count occurrences of each predicted class
        class_count = Counter(predictions)

        # Calculate total number of predicted frames
        total_predictions = sum(class_count.values())

        # Calculate the percentage for each class
        class_percentages = {cls: (count / total_predictions) * 100 for cls, count in class_count.items()}

        # Find the class with the highest percentage
        max_class = max(class_percentages, key=class_percentages.get)

        # Find a frame that corresponds to this class
        max_class_frame = next(frame['frame'] for frame in results if frame['prediction'] == max_class)

        # Return the selected frame with the highest class and the percentages
        return jsonify({
            'class_percentages': class_percentages,
            'frame_with_highest_class': f"/frames/{max_class_frame}"
        })

    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    

@app.route('/weather')
def weather():
    return render_template('wtr.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dash.html')

@app.route('/api/yearly_data')
def yearly_data():
    year = request.args.get('year')
    if year in crop_damage_data:
        monthly_data = crop_damage_data[year]
        result = {
            "labels": [],
            "damaged": [],
            "healthy": []
        }
        
        for month, weeks in monthly_data.items():
            total_diseased = 0
            total_healthy = 0
            
            for week, data in weeks.items():
                total_diseased += data["Diseased"]
                total_healthy += data["Healthy"]
            
            result["labels"].append(month)
            result["damaged"].append(total_diseased)
            result["healthy"].append(total_healthy)
        
        return jsonify(result)
    else:
        return jsonify({'error': 'Year not found'}), 404
    
@app.route('/api/monthly_data')
def monthly_data():
    year = request.args.get('year')
    month = request.args.get('month')

    if year in crop_damage_data and int(month) > 0 and int(month) <= 12:
        month_name = list(crop_damage_data[year].keys())[int(month) - 1]  # Get month name
        weeks = crop_damage_data[year][month_name]

        result = {
            "labels": [],
            "damaged": [],
            "healthy": [],
            "month": month_name  # Include month name in result
        }

        for week, data in weeks.items():
            result["labels"].append(week)
            result["damaged"].append(data["Diseased"])
            result["healthy"].append(data["Healthy"])

        return jsonify(result)
    else:
        return jsonify({'error': 'Year or month not found'}), 404



# Get input image from client, then predict class and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Get input
        filename = file.filename        
        print("@@ Input posted =", filename)
        
        file_path = os.path.join('static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        # Load image using OpenCV
        image = cv2.imread(file_path)
        
        pred, confidence = predict_image_class(model, image, class_indices)

        # Get treatment solution and cause
        treatment_solution = treatment_solutions.get(pred, "No treatment available.")
        disease_cause = disease_causes.get(pred, "No cause information available.")

        return render_template('result.html', 
                               pred_output=pred, 
                               confidence=confidence, 
                               treatment=treatment_solution, 
                               cause=disease_cause, 
                               user_image=file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080)
