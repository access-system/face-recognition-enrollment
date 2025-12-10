import threading
import time

import cv2
import mediapipe as mp

from src.blackboard import BlackboardStateful

BaseOptions = mp.tasks.BaseOptions
FaceAlignerOptions = mp.tasks.vision.FaceAlignerOptions
FaceAligner = mp.tasks.vision.FaceAligner


class FaceAlignment(BlackboardStateful):
    def __init__(self, stop_event, run_state_event, log, fps = 30):
        super().__init__()

        self.run_state_event = run_state_event
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        # Initialize MediaPipe Face Aligner
        self.landmarker_model_path = "models/face_landmarker.task"
        self.init_face_aligner()

    def start(self):
        threading.Thread(target=self.alignment_loop, daemon=True).start()

    def alignment_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping alignment.")
                break

            if not self.run_state_event.is_set():
                self.set_state("aligned_face", None)

                time.sleep(min(frame_time, 0.01))
                continue

            t1 = time.time()

            face_roi = self.get_state("detected_face")

            if face_roi is None:
                self.set_state("aligned_face", None)

                time.sleep(min(frame_time, 0.01))
                continue

            # Align face
            aligned_face = self.align_face(face_roi)

            if aligned_face is not None:
                self.set_state("aligned_face", aligned_face)
            else:
                self.set_state("aligned_face", None)

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

    def init_face_aligner(self):
        with open(self.landmarker_model_path, 'rb') as f:
            model_data = f.read()
        base_options = BaseOptions(model_asset_buffer=model_data)
        options = FaceAlignerOptions(base_options=base_options)
        self.face_aligner = FaceAligner.create_from_options(options)

    # Align face using MediaPipe Face Aligner
    def align_face(self, face_roi):
        try:
            # Convert to RGB for MediaPipe
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=face_roi_rgb
            )

            # Align face
            aligned_face = self.face_aligner.align(mp_image)

            if aligned_face is not None:
                # Convert back to BGR
                return cv2.cvtColor(aligned_face.numpy_view(), cv2.COLOR_RGB2BGR)
            return None

        except:
            return None
