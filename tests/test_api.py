import pytest
from fastapi.testclient import TestClient
from src.main import app
import os
import numpy as np
from src.predictor import GunDetector, match_gun_bbox, annotate_segmentation
from src.models import Segmentation, PredictionType

client = TestClient(app)

@pytest.fixture
def image_file():
    img_path = "gun1.jpg"
    with open(img_path, "rb") as img:
        yield img

# Test para el endpoint /annotate
def test_annotate_endpoint(image_file):
    response = client.post("/annotate", files={"file": ("gun1.jpg", image_file, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

# Test para el endpoint /detect_guns
def test_detect_guns_endpoint(image_file):
    response = client.post("/detect_guns", files={"file": ("gun1.jpg", image_file, "image/jpeg")}, data={"threshold": 0.5})
    assert response.status_code == 200
    assert "n_detections" in response.json()

# Test para el endpoint /detect_people
def test_detect_people_endpoint(image_file):
    response = client.post("/detect_people", files={"file": ("gun1.jpg", image_file, "image/jpeg")}, data={"threshold": 0.5})
    assert response.status_code == 200
    segmentation = response.json()
    assert "n_detections" in segmentation
    assert "boxes" in segmentation
    assert "labels" in segmentation

# Test para el endpoint /guns
def test_guns_endpoint(image_file):
    response = client.post("/guns", files={"file": ("gun1.jpg", image_file, "image/jpeg")}, data={"threshold": 0.5})
    assert response.status_code == 200
    guns = response.json()
    assert isinstance(guns, list)
    for gun in guns:
        assert "gun_type" in gun
        assert "location" in gun

# Test para el endpoint /people
def test_people_endpoint(image_file):
    response = client.post("/people", files={"file": ("gun1.jpg", image_file, "image/jpeg")}, data={"threshold": 0.5})
    assert response.status_code == 200
    people = response.json()
    assert isinstance(people, list)
    for person in people:
        assert "person_type" in person
        assert "location" in person
        assert "area" in person



# Test para match_gun_bbox
def test_match_gun_bbox():
    segment = [50, 50, 100, 100]  
    bboxes = [
        [200, 200, 250, 250],  # Arma 1: muy lejos
        [75, 75, 125, 125],    # Arma 2: más cercana
    ]
    max_distance = 100

    matched_box = match_gun_bbox(segment, bboxes, max_distance)

    assert matched_box == [75, 75, 125, 125]

def test_match_gun_bbox_no_match():
    segment = [50, 50, 100, 100]  
    bboxes = [
        [200, 200, 250, 250],  # Arma 1: muy lejos
    ]
    max_distance = 30

    matched_box = match_gun_bbox(segment, bboxes, max_distance)

    assert matched_box is None

# Test para segment_people
def test_segment_people():
    detector = GunDetector()
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    segmentation = detector.segment_people(image, threshold=0.5, max_distance=50)
    assert segmentation.n_detections == 0
    assert len(segmentation.boxes) == 0
    assert len(segmentation.polygons) == 0

# Test para annotate_segmentation
def test_annotate_segmentation():
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    segmentation = Segmentation(
        pred_type=PredictionType.segmentation,
        n_detections=1,
        polygons=[[[50, 50], [100, 50], [100, 100], [50, 100]]],
        boxes=[[50, 50, 100, 100]],
        labels=["danger"]
    )

    # Anotar la imagen con el segmento definido
    annotated_image = annotate_segmentation(image, segmentation, draw_boxes=True)

    # Verificar si la imagen anotada tiene los cambios realizados
    region_color = annotated_image[75, 75]
    expected_color = [0, 0, 102]
    assert np.allclose(region_color, expected_color, atol=10), f"El color de la región no es el esperado. Color actual: {region_color}"