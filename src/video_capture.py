import threading
import time

import cv2

from src.blackboard import BlackboardStateful


class VideoCapture(BlackboardStateful):
    def __init__(self, stop_event, log, fps = 30):
        super().__init__()

        self.stop_event = stop_event
        self.log = log

        self.fps = fps

    def start(self):
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def stop(self):
        self.stop_event.set()

    def capture_loop(self):
        cap = cv2.VideoCapture(0)
        frame_time = 1.0 / self.fps

        while cap.isOpened():
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping video capture.")
                break

            t1 = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            # Put frame into output
            self.set_state("default_frame", frame)

            elapsed_time = time.time() - t1
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)
