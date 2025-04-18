import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt

# Load trained models and transformers
alex_model = load_model('alexnet_trained.h5')     # AlexNet model
vgg_model = load_model('vgg16_base.h5')           # VGG16 model
scaler = joblib.load('scaler.pkl')                # Scaler
pca = joblib.load('pca.pkl')                      # PCA
classifier = joblib.load('rf_model.pkl')          # Random Forest classifier

# Alzheimer’s categories
categories = ['EMCI', 'LMCI', 'MCI', 'CN', 'AD']

# Extract hypercolumns from VGG16 and AlexNet
def extract_hypercolumn(image):
    vgg_feat = vgg_model.predict(image, verbose=0)
    alex_feat = alex_model.predict(image, verbose=0)
    return np.concatenate([vgg_feat.flatten(), alex_feat.flatten()])

# File dialog to choose image
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image")

# Preprocess image
img = cv2.imread(file_path)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized / 255.0
input_image = img_normalized.reshape(1, 224, 224, 3)

# Feature extraction, scaling and PCA
features = extract_hypercolumn(input_image)
features_scaled = scaler.transform([features])
features_pca = pca.transform(features_scaled)

# Prediction
pred_probs = classifier.predict_proba(features_pca)[0]
pred_class = np.argmax(pred_probs)
confidence = pred_probs[pred_class] * 100

# Show image and prediction
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"Prediction: {categories[pred_class]} ({confidence:.2f}%)")
plt.show()
