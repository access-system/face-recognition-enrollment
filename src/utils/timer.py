import threading
import time


def timer(fps, stop_event):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def run():
                frame_time = 1.0 / fps

                while True:
                    t1 = time.time()

                    if stop_event.is_set():
                        break

                    func(*args, **kwargs)

                    elapsed_time = time.time() - t1
                    sleep_time = max(0.0, frame_time - elapsed_time)
                    time.sleep(sleep_time)

            threading.Thread(target=run, daemon=True).start()
        return wrapper
    return decorator
