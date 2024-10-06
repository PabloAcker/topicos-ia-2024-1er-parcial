from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()

def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    segment_box = box(*segment)

    closest_gun_box = None
    min_distance = float('inf')

    for bbox in bboxes:
        gun_box = box(*bbox)
        distance = segment_box.distance(gun_box)
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            closest_gun_box = bbox

    return closest_gun_box


def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (0, 0, 255)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    annotated_image = image_array.copy()
    for polygon, box, label in zip(segmentation.polygons, segmentation.boxes, segmentation.labels):
        if label == "danger":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        overlay = annotated_image.copy()
        polygon_points = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [polygon_points], color)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

    return annotated_image



class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        seg_results = self.seg_model(image_array, conf=threshold)[0]
        person_indexes = [i for i, cls in enumerate(seg_results.boxes.cls.tolist()) if cls == 0]
        people_boxes = [
            [int(v) for v in seg_results.boxes.xyxy[i].tolist()] for i in person_indexes
        ]
        people_polygons = [
            [[int(coord) for coord in point] for point in seg_results.masks.xy[i].tolist()] for i in person_indexes
        ]
        od_results = self.od_model(image_array, conf=threshold)[0]
        gun_indexes = [
            i for i in range(len(od_results.boxes.cls.tolist())) if od_results.boxes.cls[i] in [3, 4]
        ]
        gun_boxes = [
            [int(v) for v in od_results.boxes.xyxy[i].tolist()] for i in gun_indexes
        ]
        labels = []
        for person_box in people_boxes:
            closest_gun_bbox = match_gun_bbox(person_box, gun_boxes, max_distance)
            if closest_gun_bbox is not None:
                labels.append("danger")
            else:
                labels.append("safe")
        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(people_boxes),
            polygons=people_polygons,
            boxes=people_boxes,
            labels=labels
        )