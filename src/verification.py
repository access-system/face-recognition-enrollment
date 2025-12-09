import secrets
import threading
import time

from api.access_system import validate_embedding, add_embedding


class FaceVerification:
    def __init__(self, stop_event, run_state_event, lock, shared_embedding, log, fps = 30):
        self.run_state_event = run_state_event
        self.stop_event = stop_event
        self.log = log

        self.fps = fps

        self.lock = lock
        self.shared_embedding = shared_embedding

    def start(self):
        threading.Thread(target=self.verification_loop, daemon=True).start()

    def verification_loop(self):
        frame_time = 1.0 / self.fps

        while True:
            if self.stop_event.is_set():
                self.log.info("Stop event set. Stopping validation.")
                break

            if not self.run_state_event.is_set():
                time.sleep(min(frame_time, 0.01))
                continue

            t1 = time.time()

            with self.lock:
                shared_embedding = self.shared_embedding['default']

            if shared_embedding is None:
                time.sleep(min(frame_time, 0.01))
                continue

            exists, msg = validate_embedding(shared_embedding)

            if not exists:
                status_code = add_embedding(shared_embedding, secrets.token_hex(8))

                if status_code == 201:
                    self.log.info("Embedding added successfully.")
                else:
                    self.log.info(f"Failed to add embedding. Status code: {status_code}")

                self.run_state_event.clear()
            else:
                self.log.info("Embedding already exists.")
                self.run_state_event.clear()

            elapsed_time = time.time() - t1
            # self.log.info(f"{elapsed_time:.3f} seconds per frame")
            sleep_time = max(0.0, frame_time - elapsed_time)
            time.sleep(sleep_time)
