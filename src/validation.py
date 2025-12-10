import threading
import time

import numpy as np
import cv2
import openvino as ov

from src.blackboard import BlackboardStateful


class FaceValidation(BlackboardStateful):
    def __init__(self, stop_event, run_state_event, log, fps=30, device='CPU'):
        super().__init__()

        self.stop_event = stop_event
        self.run_state_event = run_state_event

        self.log = log

        self.fps = fps

        self.core = ov.Core()
        self.device = device
        self.init_model()

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

            face_roi = self.get_state("detected_face")

            if face_roi is None:
                self.set_state("validated_face", None)

                time.sleep(min(frame_time, 0.01))
                continue

            glare, glare_msg = glare_detection(face_roi)

            if not glare:
                yaw, pitch, roll = self.estimate_head_pose(face_roi)

                if abs(yaw) < 30.0 and abs(pitch) < 30.0 and abs(roll) < 30.0:
                    self.set_state("validated_face", face_roi)
                else:
                    self.set_state("validated_face", None)
                    self.run_state_event.clear()
            else:
                self.set_state("validated_face", None)
                self.run_state_event.clear()

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
        frame = cv2.resize(frame, (h, w), interpolation=cv2.INTER_LINEAR)
        # Scale to [0, 1]
        frame = frame.astype(np.float32)
        # Change data layout from HWC to CHW
        frame = frame.transpose(2, 0, 1)
        # Add batch dimension
        frame = frame.reshape((n, c, h, w))

        return frame


def glare_detection(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Hotspots
    white_mask = gray > 230
    glare_ratio = np.sum(white_mask) / gray.size
    if glare_ratio > 0.05:
        return True, f"Glare: {glare_ratio*100:.1f}% white pixels"

    # Specular
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    specular_mask = (v_channel > 220) & (s_channel < 50)
    if np.sum(specular_mask) / gray.size > 0.03:
        return True, "High specular highlights"

    return False, "No glare"
