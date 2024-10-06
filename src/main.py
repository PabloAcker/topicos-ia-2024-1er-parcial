import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.config import get_settings
from src.models import Gun, Person, PixelLocation, PersonType, GunType

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold)
    annotated_img = annotate_segmentation(img, segmentation, draw_boxes)

    _, image_encoded = cv2.imencode('.jpg', annotated_img)
    return Response(content=image_encoded.tobytes(), media_type="image/jpeg")


@app.post("/detect")
def detect(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    detection, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold)
    return {
        "detection": detection,
        "segmentation": segmentation,
    }


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    draw_boxes: bool = True,
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold)
    annotated_img = annotate_detection(img, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes)

    _, image_encoded = cv2.imencode('.jpg', annotated_img)
    return Response(content=image_encoded.tobytes(), media_type="image/jpeg")


@app.post("/guns")
def detect_guns_info(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    detection, _ = detect_uploadfile(detector, file, threshold)
    guns = []
    
    for label, box in zip(detection.labels, detection.boxes):
        print(f"Label detected: {label}")
        if label.lower() in ["pistol", "rifle"]:
            x1, y1, x2, y2 = box
            gun_center = PixelLocation(x=int((x1 + x2) / 2), y=int((y1 + y2) / 2))
            gun_type = GunType.pistol if label.lower() == "pistol" else GunType.rifle
            guns.append(Gun(gun_type=gun_type, location=gun_center))
    return guns



@app.post("/people")
def detect_people_info(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold)

    people = []
    for label, box, polygon in zip(segmentation.labels, segmentation.boxes, segmentation.polygons):
        x1, y1, x2, y2 = box
        person_center = PixelLocation(x=int((x1 + x2) / 2), y=int((y1 + y2) / 2))
        area = int(cv2.contourArea(np.array(polygon, dtype=np.int32)))
        person_type = PersonType.safe if label == "safe" else PersonType.danger
        people.append(Person(person_type=person_type, location=person_center, area=area))
    return people

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
