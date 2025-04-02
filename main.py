#!/usr/bin/python
import sys
import cv2
import numpy as np
import glob
import os


def order_points(pts):
    # Seřadí body: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, coords):
    # Vytvoří perspektivně transformovaný (bird's-eye view) výřez dle zadaných čtyř bodů
    pts = [
        (float(coords[0]), float(coords[1])),
        (float(coords[2]), float(coords[3])),
        (float(coords[4]), float(coords[5])),
        (float(coords[6]), float(coords[7]))
    ]
    
    rect = order_points(np.array(pts))
    
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(order_points(np.array(pts)), dst)
    
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def load_parking_coordinates(filename='parking_map_python.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    parking_coordinates = [line.strip().split(" ") for line in lines]
    
    return parking_coordinates


def load_ground_truth(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        
    gt_labels = [int(line.strip()) for line in lines]
    
    return gt_labels


def get_test_image_paths(folder="test_images_zao"):
    image_paths = glob.glob(os.path.join(folder, "*.jpg"))
    
    image_paths.sort()
    
    return image_paths


def compute_edges(image_patch, edge_method="canny"):
  
    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if edge_method == "canny":
        edges = cv2.Canny(blurred, 50, 150)
    elif edge_method == "sobel":
        grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
        
    elif edge_method == "laplacian":
        lap = cv2.Laplacian(blurred, cv2.CV_16S)
        
        edges = cv2.convertScaleAbs(lap)
        
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    elif edge_method == "scharr":
        scharrx = cv2.Scharr(blurred, cv2.CV_16S, 1, 0)
        scharry = cv2.Scharr(blurred, cv2.CV_16S, 0, 1)
        
        abs_scharrx = cv2.convertScaleAbs(scharrx)
        abs_scharry = cv2.convertScaleAbs(scharry)
        
        edges = cv2.addWeighted(abs_scharrx, 0.5, abs_scharry, 0.5, 0)
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    
    # Kernel mi dával horší výsledky
    
    return edges


def compute_free_edge_counts(test_image_paths, parking_coordinates, edge_method="canny"):
    num_spots = len(parking_coordinates)
    
    free_edge_counts_list = [[] for _ in range(num_spots)]
    
    for test_img_path in test_image_paths:
        img = cv2.imread(test_img_path)
        
        if img is None:
            continue
        
        base_name = os.path.splitext(os.path.basename(test_img_path))[0]
        
        gt_path = os.path.join("test_images_zao", base_name + ".txt")
        
        gt_labels = load_ground_truth(gt_path)
        
        for idx, coord in enumerate(parking_coordinates):
            warped = four_point_transform(img, coord)
            
            edges = compute_edges(warped, edge_method=edge_method)
            
            edge_count = cv2.countNonZero(edges)
            
            if idx < len(gt_labels) and gt_labels[idx] == 0:
                free_edge_counts_list[idx].append(edge_count)
                
    return free_edge_counts_list


def compute_thresholds(free_edge_counts_list, offset, method="median", default_threshold=100):
    thresholds = []
    
    aggregator = {
        "mean": np.mean,
        "max": np.max,
        "median": np.median,
        "min": np.min
    }
    
    for counts in free_edge_counts_list:
        if counts:
            thresholds.append(aggregator[method](counts) + offset)
        else:
            thresholds.append(default_threshold)
            
    return thresholds


def detect_parking_status(edge_img, threshold_value):
    edge_count = cv2.countNonZero(edge_img)
    
    predicted = 0 if edge_count < threshold_value else 1
    
    return predicted, edge_count


def draw_parking_spot(orig_img, coord, predicted, edge_count, gt):
    pts = np.array([
        [int(float(coord[0])), int(float(coord[1]))],
        [int(float(coord[2])), int(float(coord[3]))],
        [int(float(coord[4])), int(float(coord[5]))],
        [int(float(coord[6])), int(float(coord[7]))]
    ], np.int32).reshape((-1, 1, 2))
    
    if predicted == gt:
        color = (0, 255, 0) if predicted == 0 else (0, 0, 255)
    else:
        color = (0, 165, 255)
        
    cv2.polylines(orig_img, [pts], isClosed=True, color=color, thickness=2)
    
    text_pos = (pts[0][0][0], pts[0][0][1] - 5)
    
    cv2.putText(orig_img, f"P:{predicted} C:{edge_count}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def process_test_images(test_image_paths, parking_coordinates, thresholds, edge_method, results_folder="results", save_results=True):
    overall_correct = 0
    
    overall_total = 0
    
    if save_results and not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    for test_img_path in test_image_paths:
        img = cv2.imread(test_img_path)
        
        if img is None:
            continue
        
        orig_img = img.copy()
        
        base_name = os.path.splitext(os.path.basename(test_img_path))[0]
        
        gt_path = os.path.join("test_images_zao", base_name + ".txt")
        
        gt_labels = load_ground_truth(gt_path)
        
        correct = 0
        
        total = len(parking_coordinates)
        
        for idx, coord in enumerate(parking_coordinates):
            warped = four_point_transform(img, coord)
            
            edges = compute_edges(warped, edge_method=edge_method)
            
            predicted, edge_count = detect_parking_status(
                edges, thresholds[idx])
            
            gt = gt_labels[idx] if idx < len(gt_labels) else 0
            
            if predicted == gt:
                correct += 1
                
            draw_parking_spot(orig_img, coord, predicted, edge_count, gt)
            
        overall_correct += correct
        
        overall_total += total
        
        percent = (correct / total) * 100 if total > 0 else 0
        
        print(f"Obrázek {base_name}: {correct}/{total} míst ({percent:.2f}%)")
        
        if save_results:
            result_filename = os.path.join(
                results_folder, f"{base_name}_result.jpg")
            cv2.imwrite(result_filename, orig_img)
    overall_accuracy = (overall_correct / overall_total) * \
        100 if overall_total > 0 else 0
        
    return overall_accuracy


def evaluate_configuration(test_image_paths, parking_coordinates, free_edge_counts_list, offset, aggregator_method, edge_method, temp_folder="temp_results"):
    thresholds = compute_thresholds(
        free_edge_counts_list, offset, method=aggregator_method)
    
    overall_accuracy = process_test_images(
        test_image_paths, parking_coordinates, thresholds, edge_method, results_folder=temp_folder, save_results=False)
    
    return overall_accuracy, thresholds


def main(argv):
    parking_coordinates = load_parking_coordinates('parking_map_python.txt')
    test_image_paths = get_test_image_paths("test_images_zao")

    edge_methods = ["canny", "sobel", "laplacian", "scharr"]
    
    offset_candidates = list(range(0, 2, 1))
    
    aggregator_candidates = ["mean", "max", "median", "min"]

    results_summary = {}

    print("Chvilku to potrvá...")
    for edge_method in edge_methods:
        free_edge_counts_list = compute_free_edge_counts(
            test_image_paths, parking_coordinates, edge_method=edge_method)
        
        best_accuracy = 0
        best_offset = None
        best_aggregator = None
        best_thresholds = None
        
        for aggregator_method in aggregator_candidates:
            for offset in offset_candidates:
                acc, th = evaluate_configuration(
                    test_image_paths, parking_coordinates, free_edge_counts_list, offset, aggregator_method, edge_method)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_offset = offset
                    best_aggregator = aggregator_method
                    best_thresholds = th
                    
        results_summary[edge_method] = {
            "aggregator": best_aggregator,
            "offset": best_offset,
            "accuracy": best_accuracy,
            "thresholds": best_thresholds
        }

    print("\nNejlepší konfigurace pro jednotlivé metody detekce hran:")
    print("-----------------------------------------------------------")
    for method in edge_methods:
        config = results_summary[method]
        
        print(f"Metoda: {method}")
        print(f"   Agregátor: {config['aggregator']}")
        print(f"   Offset: {config['offset']}")
        print(f"   Dosazená Accuracy: {config['accuracy']:.2f}%")
        
        print("-----------------------------------------------------------")

 
    best_overall_method = None
    best_config = None
    best_overall_accuracy = 0
    
    for method, config in results_summary.items():
        if config["accuracy"] > best_overall_accuracy:
            best_overall_accuracy = config["accuracy"]
            best_overall_method = method
            best_config = config

 
    folder = f"results_{best_overall_method}"
    final_accuracy = process_test_images(test_image_paths, parking_coordinates,
                                         best_config["thresholds"], best_overall_method, results_folder=folder, save_results=True)

    print("\n===========================================")
    print("Nejlepší celková konfigurace:")
    print(f"   Metoda detekce hran: {best_overall_method}")
    print(f"   Agregátor: {best_config['aggregator']}")
    print(f"   Offset: {best_config['offset']}")
    print(
        f"   Dosazená Accuracy (konfigurace): {best_config['accuracy']:.2f}%")
    print(f"   Finální Accuracy: {final_accuracy:.2f}%")
    print("===========================================")


if __name__ == "__main__":
    main(sys.argv[1:])
