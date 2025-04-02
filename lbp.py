#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skimage.feature import local_binary_pattern, hog
import joblib


def load_parking_coordinates(filename="parking_map_python.txt"):
    with open(filename, "r") as f:
        return [line.strip().split() for line in f.readlines()]


def load_ground_truth(gt_path):
    with open(gt_path, 'r') as f:
        return [int(line.strip()) for line in f.readlines()]


def get_test_image_paths(folder="test_images_zao"):
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_parking_spot(img, coords):
    pts = [(float(coords[i]), float(coords[i + 1])) for i in range(0, 8, 2)]
    rect = order_points(pts)
    width = max(int(np.linalg.norm(rect[2] - rect[3])), int(np.linalg.norm(rect[1] - rect[0])))
    height = max(int(np.linalg.norm(rect[1] - rect[2])), int(np.linalg.norm(rect[0] - rect[3])))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))


def extract_features(gray_img):
    gray_resized = cv2.resize(gray_img, (64, 64))
    lbp = local_binary_pattern(gray_resized, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    hog_feat = hog(gray_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                   orientations=9, block_norm='L2-Hys', visualize=False, feature_vector=True)

    mean_intensity = np.mean(gray_resized)
    std_intensity = np.std(gray_resized)
    intensity_feat = np.array([mean_intensity, std_intensity])

    return np.concatenate([lbp_hist, hog_feat, intensity_feat])


def process_dataset(image_paths, parking_coords):
    X, y = [], []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_path = img_path.replace(".jpg", ".txt")
        gt_labels = load_ground_truth(gt_path)
        for idx, coords in enumerate(parking_coords):
            if idx >= len(gt_labels):
                continue
            patch = warp_parking_spot(gray, coords)
            features = extract_features(patch)
            X.append(features)
            y.append(gt_labels[idx])
    return np.array(X), np.array(y)


def main():
    coords = load_parking_coordinates("parking_map_python.txt")
    images = get_test_image_paths("test_images_zao")

    print("\n📦 Načítám a extrahuji příznaky...")
    X, y = process_dataset(images, coords)
    X, y = shuffle(X, y, random_state=42)

    print("🔄 Normalizuji a snižuji dimenzi (PCA)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    print("\n🚀 Trénuji klasifikátory...")
    classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)),
        ("SVM (C=0.1, linear)", SVC(C=0.1, kernel='linear', gamma='scale', random_state=42)),
        ("SVM (C=1.0, rbf)", SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)),
        ("SVM (C=10, rbf)", SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5, weights='uniform')),
        ("Logistic Regression", LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=42))
    ]

    results = []
    for name, model in classifiers:
        print(f"\n🧪 Trénuji {name}...")
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} přesnost: {acc*100:.2f}%")
        print("📉 Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        joblib.dump(model, f"model_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '').lower()}.pkl")
        results.append((name, acc))

    print("\n📈 Shrnutí výsledků:")
    for i, (name, acc) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), 1):
        print(f"{i}. {name}: {acc*100:.2f}%")

    print("\n📊 Uloženo: modely + PCA + scaler")
    joblib.dump(pca, "pca_transform.pkl")
    joblib.dump(scaler, "scaler.pkl")


if __name__ == "__main__":
    main()