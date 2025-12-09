import threading
import time

import numpy as np
import cv2
import openvino as ov
from brisque import BRISQUE


class FaceValidation:
    def __init__(self, stop_event, lock, shared_frames, shared_face, log, fps=30, device='CPU'):
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        self.core = ov.Core()
        self.device = device
        self.init_model()

        self.lock = lock
        self.shared_frames = shared_frames
        self.shared_face = shared_face

        self.brisque = BRISQUE(url=False)

    def init_model(self):
        self.hpea_model_path = "models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
        self.hpea_model = self.core.read_model(model=self.hpea_model_path)
        self.hpea_compiled = self.core.compile_model(model=self.hpea_model,
                                                     device_name=self.device)

        self.input_port = self.hpea_compiled.input(0)
        self.output_y = self.hpea_compiled.output('angle_y_fc')
        self.output_p = self.hpea_compiled.output('angle_p_fc')
        self.output_r = self.hpea_compiled.output('angle_r_fc')

    def start(self):
        threading.Thread(target=self.validation_loop, daemon=True).start()

    def validation_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping validation.")
                break

            t1 = time.time()

            with self.lock:
                face_roi = self.shared_face['detected']

            if face_roi is None:
                with self.lock:
                    self.shared_face['validated'] = None

                time.sleep(min(frame_time, 0.01))
                continue

            score = self.brisque.score(face_roi)

            if score > 50:
                yaw, pitch, roll = self.estimate_head_pose(face_roi)

                self.log.info(f"BRISQUE score: {score:.2f}, Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")

                if abs(yaw) < 15 and abs(pitch) < 15 and abs(roll) < 15:
                    with self.lock:
                        self.shared_face['validated'] = face_roi
                else:
                    with self.lock:
                        self.shared_face['validated'] = None
            else:
                with self.lock:
                    self.shared_face['validated'] = None

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

    def estimate_head_pose(self, face_image):
        preprocessed_image = self.preprocess_hpea(face_image)

        # Perform inference
        result = self.hpea_compiled({self.input_port: preprocessed_image})

        # Extract yaw, pitch, roll
        yaw = float(result[self.output_y][0][0])
        pitch = float(result[self.output_p][0][0])
        roll = float(result[self.output_r][0][0])

        return yaw, pitch, roll

    def preprocess_hpea(self, frame):
        n, c, h, w = self.input_port.shape

        # Resize
        frame = cv2.resize(frame, (h, w))
        # Scale to [0, 1]
        frame = frame.astype(np.float32)
        # Change data layout from HWC to CHW
        frame = frame.transpose(2, 0, 1)
        # Add batch dimension
        frame = frame.reshape((n, c, h, w))

        return frame
