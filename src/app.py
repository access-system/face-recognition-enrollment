import flet as ft

from src.utils.timer import timer
from src.utils.converters import frame_to_base64


class EnrollmentGUI:
    def __init__(self, lock, shared_frames, stop_event, fps=30):
        self.stop_event = stop_event
        self.fps = fps

        self.lock = lock
        self.shared_frames = shared_frames

        self.placeholder = ft.Container(
            width=640,
            height=480,
            bgcolor=ft.Colors.GREY_100,
            content=ft.Row(
                controls=[ft.Icon(ft.Icons.CAMERA_ALT, size=100, color=ft.Colors.GREY_600)],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        self.image = ft.Image(
            src_base64="",
            width=640,
            height=480,
            fit=ft.ImageFit.CONTAIN,
        )

        self.frame = ft.Stack(
            controls=[self.placeholder, self.image],
            width=640,
            height=480,
        )

    def app(self, page: ft.Page):
        page.title = "Enrollment GUI"
        page.on_close = lambda e: self.stop_event.set()

        page.add(
            ft.Row(
                [
                    self.frame,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            )
        )

        @timer(self.fps, self.stop_event)
        def update_frame():
            with self.lock:
                default_frame = self.shared_frames["default"]

            has_frame = default_frame is not None

            # Toggle visibility
            self.image.visible = has_frame
            self.placeholder.visible = not has_frame

            if has_frame:
                self.image.src_base64 = frame_to_base64(default_frame)

            self.image.update()
            self.placeholder.update()

        update_frame()
