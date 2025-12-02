import time

import cv2


class VideoStream:
    def __init__(self, stop_event, lock, shared_frames, log, name: str = "Face Detection", fps = 30):
        self.stop_event = stop_event
        self.log = log

        self.name = name
        self.fps = fps

        self.lock = lock
        self.shared_frames = shared_frames

    def start(self):
        self.stream_loop()

    def stream_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            # Check for stop event
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping video stream.")
                break

            t1 = time.time()

            # Get latest and processed frames
            with self.lock:
                processed_frame = self.shared_frames['processed']
                default_frame = self.shared_frames['default']

            if default_frame is None:
                continue

            # Display processed frame if available, else display latest frame
            if processed_frame is not None:
                cv2.imshow(self.name, cv2.flip(processed_frame, 1))
            else:
                cv2.imshow(self.name, cv2.flip(default_frame, 1))

            elapsed_time = time.time() - t1
            # self.log.info(f"{elapsed_time:.3f} seconds per frame")
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
