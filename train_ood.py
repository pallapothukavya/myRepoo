import os
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
import pickle
import joblib # to save model

train_dir = r'media\main_dataset\train'
features = []

print("Extracting features from training set...")
for root, _, files in os.walk(train_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, f)
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((160, 120))
                arr = np.array(img) / 255.0
                mean_rgb = np.mean(arr, axis=(0, 1))
                std_rgb = np.std(arr, axis=(0, 1))
                feature_vector = np.concatenate([mean_rgb, std_rgb])
                features.append(feature_vector)
            except Exception as e:
                pass
            
            if len(features) % 1000 == 0:
                print(f"Processed {len(features)} images...")

X_train = np.array(features)
print(f"Training IsolationForest on {len(X_train)} samples...")

# Train Isolation Forest
# contamination is the proportion of outliers in the dataset. Since it's all training data, we assume very low like 0.01 or 0.02
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
clf.fit(X_train)

# Save model
joblib.dump(clf, 'ood_model.pkl')
print("Model saved to ood_model.pkl")

# Test on a random black image
black = np.zeros((120, 160, 3))
black_f = np.concatenate([np.mean(black, axis=(0, 1)), np.std(black, axis=(0, 1))]).reshape(1, -1)
print("Black image prediction (-1 means anomaly, 1 means normal):", clf.predict(black_f))

# Test on a random noise image
noise = np.random.rand(120, 160, 3)
noise_f = np.concatenate([np.mean(noise, axis=(0, 1)), np.std(noise, axis=(0, 1))]).reshape(1, -1)
print("Random noise prediction:", clf.predict(noise_f))

# Test on a typical blood image from test set
test_img_path = r'media\main_dataset\test\EOSINOPHIL\_0_207.jpeg'
test_img = Image.open(test_img_path).convert('RGB').resize((160, 120))
test_arr = np.array(test_img) / 255.0
test_f = np.concatenate([np.mean(test_arr, axis=(0, 1)), np.std(test_arr, axis=(0, 1))]).reshape(1, -1)
print("Test blood image prediction:", clf.predict(test_f))
