import flet as ft

from src.utils.timer import timer
from src.utils.converters import frame_to_base64


class EnrollmentGUI:
    def __init__(self, lock, shared_frames, stop_event, fps=30):
        self.stop_event = stop_event
        self.fps = fps

        self.lock = lock
        self.shared_frames = shared_frames

    def app(self, page: ft.Page):
        page.title = "Enrollment GUI"
        page.on_close = lambda e: self.stop_event.set()

        if self.shared_frames["default"] is not None:
            self.frame = ft.Image(
                src_base64=self.shared_frames["default"],
                width=640,
                height=480,
                fit=ft.ImageFit.CONTAIN,
            )
        else:
            self.frame = ft.Image(
                src="resources/img/placeholder.jpg",
                width=640,
                height=480,
                fit=ft.ImageFit.CONTAIN,
            )

        @timer(self.fps, self.stop_event)
        def update_frame():
            with self.lock:
                default_frame = self.shared_frames["default"]

            if default_frame is not None:
                self.frame.src_base64 = frame_to_base64(default_frame)
                self.frame.update()

        update_frame()

        page.add(
            ft.Row(
                [
                    self.frame,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )
