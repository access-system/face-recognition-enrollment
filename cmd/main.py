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

from src.pipeline_manager import PipelineManager


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

    device = 'GPU'

    deps = {
        "lock": lock, "stop_event": stop_event, "run_state_event": run_state_event,
        "shared_frames": shared_frames, "shared_face": shared_face, "shared_embedding": shared_embedding,
        "log": log, "fps": fps, "device": device
    }
    classes = [VideoCapture, FaceDetection, FaceValidation, FaceAlignment, RecognitionArcFace, FaceVerification]

    pipeline_manager = PipelineManager(deps, classes)
    pipeline_manager.build()
    pipeline_manager.run()

    app = EnrollmentGUI(lock, shared_frames, stop_event, run_state_event, fps)
    ft.app(target=app.app)


if __name__ == '__main__':
    main()
