import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Define the classes
CLASS_NAMES = ["covid19", "lungopacity", "normal", "tuberculosis", "viralpneumonia"]

# Update this path to your model file location
MODEL_PATH = "C:/Users/Adityagh/OneDrive/Desktop/AIMI Project/Medical-Image-Diagnosis-using-Convolutional-Neural-Networks/alexnet.pth"

# HTML template directly in the code - no need for a separate file
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Chest X-ray Disease Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .upload-section {
            text-align: center;
            margin: 20px 0;
        }
        #upload-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        #upload-btn:hover {
            background-color: #45a049;
        }
        #result-container {
            margin-top: 20px;
            display: none;
        }
        #image-preview {
            max-width: 100%;
            max-height: 300px;
            margin: 10px auto;
            display: block;
            border: 1px solid #ddd;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .result-item:last-child {
            border-bottom: none;
        }
        .loading {
            text-align: center;
            margin: 20px 0;
            display: none;
        }
        .disease-name {
            text-transform: capitalize;
            font-weight: bold;
        }
        .probability {
            font-weight: bold;
        }
        #top-result {
            background-color: #f9f9f9;
            border-radius: 1px;
            padding: 1px;
            margin-bottom: 1px;
            text-align: center;
             color: #ffffff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chest X-ray Disease Classifier</h1>
        <p>Upload a chest X-ray image to detect respiratory conditions including COVID-19, Lung Opacity, Normal (healthy lungs), Tuberculosis, and Viral Pneumonia.</p>
        
        <div class="upload-section">
            <input type="file" id="file-input" accept="image/*" style="display: none;">
            <button id="upload-btn" onclick="document.getElementById('file-input').click()">
                Upload X-ray Image
            </button>
        </div>
        
        <div id="loading" class="loading">
            Analyzing image... Please wait.
        </div>
        
        <div id="result-container">
            <h2>Results:</h2>
            <img id="image-preview" src="" alt="Uploaded X-ray">
            <div id="top-result"></div>
            <div id="results-list"></div>
        </div>
    </div>

    <script>
        document.getElementById('file-input').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-container').style.display = 'none';
            
            // Display image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('image-preview').src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            // Create form data for upload
            const formData = new FormData();
            formData.append('file', file);
            
            // Send request to server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result-container').style.display = 'block';
                
                // Display results
                const resultsList = document.getElementById('results-list');
                resultsList.innerHTML = '';
                
                // Check for error
                if (data.error) {
                    resultsList.innerHTML = `<div class="error">${data.error}</div>`;
                    return;
                }
                
                // Get the top result (first key in sorted data)
                const topClass = Object.keys(data)[0];
                const topProb = data[topClass];
                document.getElementById('top-result').innerHTML = `
                    <h3>Detected: <span class="disease-name">${formatClassName(topClass)}</span></h3>
                    <p>Confidence: <span class="probability">${topProb}</span></p>
                `;
                
                // Add all results to the list
                for (const [className, probability] of Object.entries(data)) {
                    const item = document.createElement('div');
                    item.className = 'result-item';
                    item.innerHTML = `
                        <span class="disease-name">${formatClassName(className)}</span>
                        <span class="probability">${probability}</span>
                    `;
                    resultsList.appendChild(item);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result-container').style.display = 'block';
                document.getElementById('results-list').innerHTML = `
                    <div class="error">Error: ${error.message}</div>
                `;
            });
        });
        
        // Format class names for display
        function formatClassName(name) {
            if (name === "covid19") return "COVID-19";
            if (name === "lungopacity") return "Lung Opacity";
            if (name === "viralpneumonia") return "Viral Pneumonia";
            return name.charAt(0).toUpperCase() + name.slice(1);
        }
    </script>
</body>
</html>
"""

def load_model():
    """Load the pretrained AlexNet model"""
    try:
        model = models.alexnet()
        
        # Modify classifier to match model's output size (5 classes)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=5)
        
        # Load the state_dict
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # Set to eval mode
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_path):
    """Preprocess the input image"""
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet requires 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard for pretrained ImageNet models
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # add batch dimension

def predict(image_path):
    """Run inference on the input image"""
    try:
        # Load model if not already loaded
        if 'model' not in globals():
            global model
            model = load_model()
        
        # Process image
        input_tensor = preprocess_image(image_path)
        
        # Run through model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        # Get results for all classes
        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            probability = float(probabilities[i]) * 100
            results[class_name] = f"{probability:.2f}%"
        
        # Sort by probability (highest first)
        sorted_results = {k: v for k, v in sorted(
            results.items(), key=lambda item: float(item[1].strip('%')), reverse=True
        )}
        
        return sorted_results
        
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_path = os.path.join('uploads', file.filename)
        file.save(temp_path)
        
        # Get prediction
        prediction = predict(temp_path)
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print(f"Please update the MODEL_PATH variable in the code to point to your alexnet.pth file")
    else:
        print(f"Model found at {MODEL_PATH}")
        # Try to load the model at startup to catch any issues early
        try:
            load_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    # Start the Flask app
    print("Starting server... Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)