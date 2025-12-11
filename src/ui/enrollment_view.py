import cv2
import flet as ft

from src.blackboard import BlackboardStateful
from src.utils.converters import frame_to_base64
from src.utils.timer import timer


class EnrollmentView(ft.Row, BlackboardStateful):
    def __init__(self, stop_event, run_state_event, pipeline_manager, fps=30):
        super().__init__()

        self.stop_event = stop_event
        self.run_state_event = run_state_event

        self.pipeline_manager = pipeline_manager

        self.fps = fps

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

        self.controls = [
            self.frame,
            ft.Column(
                height=480,
                controls=[self.run_state_btn],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
        ]

        self.alignment = ft.MainAxisAlignment.CENTER
        self.horizontal_alignment = ft.MainAxisAlignment.CENTER

    def did_mount(self):
        self.stop_event.clear()

        self.pipeline_manager.run()
        self._start_frame_update()

    def will_unmount(self):
        self.stop_event.set()
        self.reset_all()

    def _start_frame_update(self):
        @timer(self.fps, self.stop_event)
        def update_frame():
            frame = self.select_frame()

            has_frame = frame is not None

            # Toggle visibility
            self.image.visible = has_frame
            self.placeholder.visible = not has_frame

            if has_frame:
                frame = cv2.flip(frame, 1)
                self.image.src_base64 = frame_to_base64(frame)

            self.image.update()
            self.placeholder.update()

        update_frame()

    def toggle_enrollment(self, e):
        if not self.run_state_event.is_set():
            self.run_state_event.set()
        else:
            self.run_state_event.clear()

        self.run_state_btn.update()

    def select_frame(self):
        default_frame = self.get_state("default_frame")
        processed_frame = self.get_state("processed_frame")

        if processed_frame is not None:
            return processed_frame
        else:
            return default_frame
