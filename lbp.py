#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import random
import time
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
    hog_feat = hog(gray_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys', visualize=False, feature_vector=True)
    mean_intensity = np.mean(gray_resized)
    std_intensity = np.std(gray_resized)
    intensity_feat = np.array([mean_intensity, std_intensity])
    return np.concatenate([lbp_hist, hog_feat, intensity_feat])

def process_dataset(image_paths, parking_coords):
    X, y = [], []
    patches = []
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
            patches.append((img_path, coords, gt_labels[idx]))
    return np.array(X), np.array(y), patches

def draw_results(best_model, pca, scaler, coords, output_folder="output_visual"):
    import shutil
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    image_paths = sorted(get_test_image_paths("test_images_zao"), key=lambda x: os.path.basename(x).lower())
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt_path = img_path.replace(".jpg", ".txt")
        gt_labels = load_ground_truth(gt_path)
        correct = 0
        total = 0
        for idx, coord in enumerate(coords):
            patch = warp_parking_spot(gray, coord)
            features = extract_features(patch)
            features_scaled = scaler.transform([features])
            features_pca = pca.transform(features_scaled)
            pred = best_model.predict(features_pca)[0]
            gt = gt_labels[idx] if idx < len(gt_labels) else 0
            if pred == gt:
                correct += 1
            total += 1
            if pred != gt:
                color = (0, 165, 255)
            elif pred == 0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            pts = np.array([(int(float(coord[i])), int(float(coord[i + 1]))) for i in range(0, 8, 2)], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(original, [pts], isClosed=True, color=color, thickness=2)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"📸 {os.path.basename(img_path)}: {correct}/{total} správně | Úspěšnost: {accuracy:.2f}%")
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_folder, out_name), original)


def run_classifier(name, model, X_pca, y):
    print(f"\n▶️ Spouštím klasifikátor: {name}")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Přesnost: {acc*100:.2f}%")
    print("📉 Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def test_lbp_configurations():
    lbp_configs = [(8, 1), (16, 2), (24, 3)]
    summary_results = []
    print("\n🔍 Spouštím experiment s různými LBP konfiguracemi:")
    coords = load_parking_coordinates("parking_map_python.txt")
    images = get_test_image_paths("test_images_zao")

    for P, R in lbp_configs:
        print(f"\n⚙️ Testuji LBP konfiguraci: P={P}, R={R}")
        start_time = time.time()

        def extract_features_lbp(gray_img):
            gray_resized = cv2.resize(gray_img, (64, 64))
            lbp = local_binary_pattern(gray_resized, P, R, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
            lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
            hog_feat = hog(gray_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                           orientations=9, block_norm='L2-Hys', visualize=False, feature_vector=True)
            mean_intensity = np.mean(gray_resized)
            std_intensity = np.std(gray_resized)
            intensity_feat = np.array([mean_intensity, std_intensity])
            return np.concatenate([lbp_hist, hog_feat, intensity_feat])

        X, y = [], []
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gt_path = img_path.replace(".jpg", ".txt")
            gt_labels = load_ground_truth(gt_path)
            for idx, coords_i in enumerate(coords):
                if idx >= len(gt_labels):
                    continue
                patch = warp_parking_spot(gray, coords_i)
                features = extract_features_lbp(patch)
                X.append(features)
                y.append(gt_labels[idx])

        X, y = shuffle(X, y, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
        model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - start_time

        print(f"✅ Přesnost: {acc*100:.2f}%")
        print(f"⏱️ Čas zpracování: {elapsed:.2f} sekund")
        summary_results.append((P, R, acc, elapsed))

    print("\n📊 Shrnutí LBP konfigurací:")
    print("P\tR\tPřesnost (%)\tČas (s)")
    for P, R, acc, t in summary_results:
        print(f"{P}\t{R}\t{acc*100:.2f}\t\t{t:.2f}")

def main():
    while True:
        print("\n🔧 Vyber klasifikátor pro ladění:")
        options = {
            "1": "SVM",
            "2": "Random Forest",
            "3": "Logistic Regression",
            "4": "K-Nearest Neighbors",
            "5": "Decision Tree",
            "6": "Statistika všech",
            "7": "Random tuning",
            "8": "LBP konfigurace test",
            "q": "Konec"
        }
        for k, v in options.items():
            print(f"{k}: {v}")
        choice = input("\nZadej číslo klasifikátoru nebo 'q' pro konec: ").strip()
        if choice.lower() == 'q':
            print("👋 Ukončuji program...")
            break

        if choice == "8":
            test_lbp_configurations()
            continue

        classifier_name = options.get(choice)
        if not classifier_name:
            print("Neplatná volba.")
            continue

        coords = load_parking_coordinates("parking_map_python.txt")
        images = get_test_image_paths("test_images_zao")
        print("\n📦 Načítám a extrahuji příznaky...")
        X, y, _ = process_dataset(images, coords)
        X, y = shuffle(X, y, random_state=42)

        print("🔄 Normalizuji a snižuji dimenzi (PCA)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        if choice == "1":
            run_classifier("SVM", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42), X_pca, y)
            continue
        elif choice == "2":
            run_classifier("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), X_pca, y)
            continue
        elif choice == "3":
            run_classifier("Logistic Regression", LogisticRegression(max_iter=300, random_state=42), X_pca, y)
            continue
        elif choice == "4":
            run_classifier("K-Nearest Neighbors", KNeighborsClassifier(), X_pca, y)
            continue
        elif choice == "5":
            run_classifier("Decision Tree", DecisionTreeClassifier(random_state=42), X_pca, y)
            continue
        elif choice == "6":
            classifiers = [
                ("SVM", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)),
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("Logistic Regression", LogisticRegression(max_iter=300, random_state=42)),
                ("K-Nearest Neighbors", KNeighborsClassifier()),
                ("Decision Tree", DecisionTreeClassifier(random_state=42))
            ]
            for name, model in classifiers:
                run_classifier(name, model, X_pca, y)
            continue

        if choice == "7":
            print("\n🎲 Spouštím náhodné ladění všech klasifikátorů:")
            classifiers = []

            for _ in range(3):
                rf_params = {
                    "n_estimators": random.randint(50, 250),
                    "max_depth": random.choice([None, random.randint(5, 30)]),
                    "min_samples_split": random.randint(2, 10)
                }
                classifiers.append(("Random Forest", RandomForestClassifier(**rf_params, random_state=42), rf_params))

                svm_params = {
                    "C": round(random.uniform(0.01, 10.0), 2),
                    "kernel": random.choice(['linear', 'rbf', 'sigmoid', 'poly']),
                    "gamma": random.choice(['scale', 'auto']),
                    "probability": True
                }
                classifiers.append(("SVM", SVC(**svm_params, random_state=42), svm_params))

                lr_params = {
                    "C": round(random.uniform(0.01, 10.0), 2),
                    "solver": random.choice(['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']),
                    "max_iter": random.choice([200, 300, 500, 1000])
                }
                classifiers.append(("Logistic Regression", LogisticRegression(**lr_params, random_state=42), lr_params))

                knn_params = {
                    "n_neighbors": random.randint(1, 15),
                    "weights": random.choice(['uniform', 'distance']),
                    "algorithm": random.choice(['auto', 'ball_tree', 'kd_tree', 'brute'])
                }
                classifiers.append(("K-Nearest Neighbors", KNeighborsClassifier(**knn_params), knn_params))

                dt_params = {
                    "max_depth": random.choice([None] + list(range(5, 25))),
                    "min_samples_split": random.randint(2, 10),
                    "criterion": random.choice(['gini', 'entropy', 'log_loss'])
                }
                classifiers.append(("Decision Tree", DecisionTreeClassifier(**dt_params, random_state=42), dt_params))

            results = []
            for name, model, params in classifiers:
                print(f"\n🧪 Trénuji {name} s parametry: {params}")
                X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"✅ Přesnost: {acc*100:.2f}%")
                print("📉 Confusion matrix:")
                print(confusion_matrix(y_test, y_pred))
                results.append((name, acc, model, params))

            best = max(results, key=lambda x: x[1])
            print("\n📊 Všechny výsledky random tuningu:")
            for i, (name, acc, _, params) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), start=1):
                print(f"{i}. {name}: {acc*100:.2f}% | Parametry: {params}")

            print("\n🏆 Nejlepší klasifikátor z random tuningu:")
            print(f"1. {best[0]}: {best[1]*100:.2f}%")
            print(f"   🔧 Parametry: {best[3]}")

            best_model = best[2]
            draw_results(best_model, pca, scaler, coords)
            print("🖼️ Vizualizace výsledků byla uložena do složky 'output_visual'.")

            try:
                with open("best.txt", "r") as f:
                    best_name = f.readline().strip()
                    best_score = float(f.readline().strip())
            except Exception:
                best_name, best_score = "Nothing", 0.0

            if best[1] * 100 > best_score:
                print(f"💾 Nový nejlepší model! Ukládám do best.txt")
                with open("best.txt", "w") as f:
                    f.write(f"{best[0]} | {best[3]}\n")
                    f.write(f"{best[1]*100:.2f}\n")

            continue




if __name__ == "__main__":
    main()
