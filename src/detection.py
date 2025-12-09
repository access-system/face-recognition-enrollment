import threading
import time

import cv2

import mediapipe as mp


class FaceDetection:
    def __init__(self, stop_event, lock, shared_frames, face, log, fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        # Initialize MediaPipe Face Detection
        self.init_face_detection()

        # Initialize MediaPipe Drawing Utils
        self.mp_drawing = mp.solutions.drawing_utils

        self.lock = lock
        self.shared_frames = shared_frames # latest, processed
        self.face = face

    def start(self):
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def detection_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping detection.")
                break

            t1 = time.time()

            with self.lock:
                default_frame = self.shared_frames['default']

            if default_frame is None:
                time.sleep(min(frame_time, 0.01))
                continue

            # Detect faces
            results = self.detect_face(default_frame)
            if results[0] is not None:
                # Make bounding boxes
                bboxes = make_bboxes(default_frame, results[0])
                if bboxes is not None:
                    face_roi = self.get_face_roi(default_frame, bboxes[0])

                    if face_roi is not None:
                        with self.lock:
                            self.face['detected'] = face_roi
                    else:
                        with self.lock:
                            self.face['detected'] = None
                else:
                    with self.lock:
                        self.face['detected'] = None

                processed_frame = self.draw_detections(default_frame, results[0])
                with self.lock:
                    self.shared_frames['processed'] = processed_frame

            else:
                with self.lock:
                    self.face['detected'] = None

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

    def init_face_detection(self):
        mp_face_detection = mp.solutions.face_detection.FaceDetection
        self.face_detection = mp_face_detection(model_selection=0, min_detection_confidence=0.5)

    # Detect face using MediaPipe Face Detection
    def detect_face(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.face_detection.process(frame)

        return detections

    def get_face_roi(self, frame, bbox):
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # Ensure coordinates are within frame bounds
        x = max(0, int(x))
        y = max(0, int(y))
        x_end = min(frame_w, int(x + w))
        y_end = min(frame_h, int(y + h))

        # Calculate ROI width and height
        roi_w = x_end - x
        roi_h = y_end - y

        # Check for valid ROI
        if roi_w <= 0 or roi_h <= 0:
            self.log.error("Invalid ROI size.")
            return None

        # Extract the region of interest
        face_roi = frame[y:y + h, x:x + w]

        # Check if the ROI is valid
        if face_roi.size == 0:
            self.log.error("Invalid ROI size.")
            return None

        return face_roi

    # Draw detections on frame
    def draw_detections(self, frame, results):
        frame.flags.writeable = True
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(frame, detection)
        return frame


# Create bounding boxes from MediaPipe detection results
def make_bboxes(frame, results):
    h, w, _ = frame.shape
    bboxes = []

    if results.detections:
        for detection in results.detections:
            bbox_c = detection.location_data.relative_bounding_box

            # Convert normalized coordinates to pixel values
            x_min = max(0.0, min(1.0, bbox_c.xmin))
            y_min = max(0.0, min(1.0, bbox_c.ymin))
            box_width = max(0.0, min(1.0, bbox_c.width))
            box_height = max(0.0, min(1.0, bbox_c.height))

            # Scale to image size
            x_min = int(x_min * w)
            y_min = int(y_min * h)
            box_width = int(box_width * w)
            box_height = int(box_height * h)

            # Ensure box dimensions are positive
            if box_width > 0 and box_height > 0:
                bboxes.append((x_min, y_min, box_width, box_height))

    return bboxes
