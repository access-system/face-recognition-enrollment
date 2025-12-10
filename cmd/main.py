import threading

import flet as ft
import loguru

from src.pipelines.video_capture import VideoCapture
from src.pipelines.detection import FaceDetection
from src.pipelines.validation import FaceValidation
from src.pipelines.alignment import FaceAlignment
from src.pipelines.recognition import RecognitionArcFace
from src.pipelines.verification import FaceVerification
from src.app import EnrollmentGUI

from src.pipeline_manager import PipelineManager


def main():
    run_state_event = threading.Event()
    stop_event = threading.Event()

    log = loguru.logger

    fps = 30
    log.info(f"Set FPS to {fps}...")

    device = 'GPU'

    deps = {
        "stop_event": stop_event, "run_state_event": run_state_event,
        "log": log, "fps": fps, "device": device
    }
    classes = [VideoCapture, FaceDetection, FaceValidation, FaceAlignment, RecognitionArcFace, FaceVerification]

    pipeline_manager = PipelineManager(deps, classes)
    pipeline_manager.build()
    pipeline_manager.run()

    app = EnrollmentGUI(stop_event, run_state_event, fps)
    ft.app(target=app.app)


if __name__ == '__main__':
    main()
