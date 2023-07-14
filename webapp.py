import argparse
import cv2
import torch
from flask import Flask, render_template, Response
from PIL import Image
from sklearn.metrics import pairwise_distances as euclidean_distances
import numpy as np
import logging

app = Flask(__name__)

# Configuration
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snow', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']
model_path = 'yolov7-tiny.pt'
max_disappeared = 40
distance_threshold = 50

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


@app.route("/")
def hello_world():
    return render_template('index.html')


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class Person:
    id_map = {}  # Dictionary to store the mapping of object ID to person information

    def __init__(self, objectID, centroid, feature):
        self.objectID = objectID
        self.centroid = centroid
        self.disappeared = 0
        self.feature = feature  # Add the feature attribute

        # Update the id_map with the object ID and person information
        self.id_map[objectID] = self


class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = {}
        self.tracked_persons = set()  # New attribute

    def register(self, centroid, feature):
        person = Person(self.nextObjectID, centroid, feature)
        self.objects[self.nextObjectID] = person
        self.nextObjectID += 1
        # Add the object ID to tracked_persons set
        self.tracked_persons.add(person.objectID)

    def deregister(self, objectID):
        del self.objects[objectID]
        if objectID in Person.id_map:
            del Person.id_map[objectID]  # Remove the entry from the id_map

    def update(self, rects, features):
        if len(rects) == 0:
            # Create a copy of the dictionary values
            for person in list(self.objects.values()):
                person.disappeared += 1
                if person.disappeared > max_disappeared:
                    self.deregister(person.objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], features[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [
                person.centroid for person in self.objects.values()]

            D = euclidean_distances(objectCentroids, inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                person = self.objects[objectID]
                person.centroid = inputCentroids[col]
                person.disappeared = 0

                usedRows.add(row)
                usedCols.add(col)

                # Add the object ID to tracked_persons set
                self.tracked_persons.add(objectID)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                person = self.objects[objectID]
                person.disappeared += 1

                if person.disappeared > max_disappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], features[col])

        return self.objects


def extract_appearance_features(model, frame, person_rects):
    features = []

    for (x1, y1, x2, y2) in person_rects:
        # Extract the region of interest (person) from the frame
        roi = frame[y1:y2, x1:x2]

        # Preprocess the ROI (e.g., resize, normalize, convert to tensor)
        roi = cv2.resize(roi, (416, 416))
        roi = roi.astype(np.float32) / 255.0
        roi = np.transpose(roi, (2, 0, 1))
        roi = np.expand_dims(roi, axis=0)

        # Convert the ROI to a PyTorch tensor
        roi = torch.from_numpy(roi)

        # Pass the ROI through the YOLOv7-tiny model and obtain the feature representation
        with torch.no_grad():
            features_tensor = model(roi.cuda())

        # Convert the features to a numpy array
        features_np = features_tensor[0].squeeze().cpu().numpy()

        # Append the appearance-based feature to the list
        features.append(features_np)

    return features


def get_features(frame, person_rects):
    features = extract_appearance_features(model, frame, person_rects)
    return features


@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the desired width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the desired height

    total_person_count = 0
    ct = CentroidTracker(maxDisappeared=max_disappeared)
    previous_persons = []

    def generate():
        nonlocal total_person_count, previous_persons

        while True:
            success, frame = cap.read()
            if not success:
                break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(img, size=640)
            results.print()

            img_BGR = frame

            person_rects = []

            for pred in results.pred:
                boxes = pred[:, :4]
                confs = pred[:, 4]
                classes = pred[:, 5]

                for box, conf, cls in zip(boxes, confs, classes):
                    if int(cls) == 0 and conf > 0.5:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        person_rects.append((x1, y1, x2, y2))
                        cv2.rectangle(img_BGR, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        class_name = class_names[int(cls)]
                        conf_text = f"{class_name}: {conf:.2f}"
                        cv2.putText(img_BGR, conf_text, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            features = get_features(frame, person_rects)

            objects = ct.update(person_rects, features)

            current_person_count = 0

            for person in objects.values():
                is_same_person = False
                for prev_person in previous_persons:
                    distance = euclidean_distance(
                        person.feature, prev_person.feature)
                    if distance < distance_threshold:
                        is_same_person = True
                        break

                if not is_same_person:
                    previous_persons.append(person)
                    total_person_count += 1

                text = f"ID: {person.objectID}"
                cv2.putText(img_BGR, text, (person.centroid[0] - 10, person.centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                    img_BGR, (person.centroid[0], person.centroid[1]), 4, (0, 255, 0), -1)
                current_person_count += 1

            cv2.putText(img_BGR, f'Current Person(s): {current_person_count}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.putText(img_BGR, f'Total Person(s): {total_person_count}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', img_BGR)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing YOLOv7 models")
    parser.add_argument("--port", default=8000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom', model_path,
                           source='local').eval().cuda()

    logger.info("Starting the application...")
    app.run(host="0.0.0.0", port=args.port)
