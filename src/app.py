import cv2
import flet as ft

from src.utils.timer import timer
from src.utils.converters import frame_to_base64


class EnrollmentGUI:
    def __init__(self, lock, shared_frames, stop_event, run_state_event, fps=30):
        self.run_state_event = run_state_event
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

        self.run_state_btn = ft.ElevatedButton(
            text="Start Enrollment",
            on_click=self.toggle_enrollment,
        )

    def toggle_enrollment(self, e):
        if not self.run_state_event.is_set():
            self.run_state_event.set()
            self.run_state_btn.text = "Stop Enrollment"
        else:
            self.run_state_event.clear()
            self.run_state_btn.text = "Start Enrollment"

        self.run_state_btn.update()

    def app(self, page: ft.Page):
        page.title = "Enrollment GUI"
        page.on_close = lambda e: self.stop_event.set()

        page.add(
            ft.Row(
                [
                    self.frame,
                    self.run_state_btn,
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
                self.image.src_base64 = frame_to_base64(self.select_frame())

            self.image.update()
            self.placeholder.update()

        update_frame()

    def select_frame(self):
        with self.lock:
            default_frame = self.shared_frames["default"]
            processed_frame = self.shared_frames["processed"]

        if processed_frame is not None:
            return processed_frame
        else:
            return default_frame
