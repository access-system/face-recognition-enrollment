import threading
import time

import cv2
import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
FaceAlignerOptions = mp.tasks.vision.FaceAlignerOptions
FaceAligner = mp.tasks.vision.FaceAligner


class FaceAlignment:
    def __init__(self, stop_event, lock, shared_frames, shared_face, log, fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        # Initialize MediaPipe Face Aligner
        self.landmarker_model_path = "models/face_landmarker.task"
        self.init_face_aligner()

        self.lock = lock
        self.shared_frames = shared_frames
        self.shared_face = shared_face

    def start(self):
        threading.Thread(target=self.alignment_loop, daemon=True).start()

    def alignment_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping alignment.")
                break

            t1 = time.time()

            with self.lock:
                face_roi = self.shared_frames['validated']

            if face_roi is None:
                with self.lock:
                    self.shared_face['aligned'] = None

                time.sleep(min(frame_time, 0.01))
                continue

            # Align face
            aligned_face = self.align_face(face_roi)

            if aligned_face is not None:
                with self.lock:
                    self.shared_face['aligned'] = aligned_face
            else:
                with self.lock:
                    self.shared_face['aligned'] = None

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
