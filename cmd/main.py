import threading

import flet as ft
import loguru

from src.video_capture import VideoCapture
from src.detection import FaceDetection
from src.validation import FaceValidation
from src.alignment import FaceAlignment
from src.recognition import RecognitionArcFace
from src.verification import FaceVerification
from src.app import EnrollmentGUI


def main():
    lock = threading.Lock()

    run_state_event = threading.Event()
    stop_event = threading.Event()

    log = loguru.logger

    shared_frames = {'default': None, 'processed': None}
    shared_face = {'detected': None, 'validated': None, 'aligned': None}
    shared_embedding = {'default': None}

    log.info("Set FPS to 20...")
    fps = 30

    log.info("Setup pipelines...")
    video_capture = VideoCapture(stop_event, lock, shared_frames, log, fps=fps)
    detection_mediapipe = FaceDetection(stop_event, lock, shared_frames, face, log, fps=fps)
    face_validation = FaceValidation(stop_event, lock, shared_frames, face, log, fps=fps)
    face_alignment = FaceAlignment(stop_event, run_state_event, lock, face, face, log, fps=fps)
    recognition_arcface = RecognitionArcFace(stop_event, run_state_event, lock, face, shared_embedding, log, device='GPU',
                                             fps=fps)
    face_verification = FaceVerification(stop_event, run_state_event, lock, shared_embedding, log, fps=fps)

    log.info("Starting pipelines...")
    video_capture.start()
    detection_mediapipe.start()
    face_validation.start()
    face_alignment.start()
    recognition_arcface.start()
    face_verification.start()

    app = EnrollmentGUI(lock, shared_frames, stop_event, run_state_event)
    ft.app(target=app.app)


if __name__ == '__main__':
    main()
