import flet as ft

from src.ui.enrollment_view import EnrollmentView


class EnrollmentGUI:
    def __init__(self, pipeline_manager, stop_event, run_state_event, fps=30):
        self.run_state_event = run_state_event
        self.stop_event = stop_event
        self.fps = fps

        self.pipeline_manager = pipeline_manager

    def main(self, page: ft.Page):
        page.title = "Enrollment GUI"
        page.on_close = lambda e: self.stop_event.set()

        def route_change(e):
            page.views.clear()
            page.views.append(self.create_home_view(page))
            if page.route == "/enrollment":
                page.views.append(self.create_enrollment_view(page))
            page.update()

        page.on_route_change = route_change
        page.go("home")

    def create_home_view(self, page):
        def go_enrollment(e):
            page.go("/enrollment")

        return ft.View(
            "/home",
            [
                ft.AppBar(title=ft.Text("Home"), bgcolor=ft.Colors.SURFACE),
                ft.ElevatedButton("Go to Enrollment", on_click=go_enrollment),
                ft.Text("Home page content")
            ]
        )

    def create_enrollment_view(self, page):
        def back(e):
            page.views.pop()
            top_view = page.views[-1]
            page.go(top_view.route)
            page.update()

        return ft.View(
            "/enrollment",
            [
                ft.AppBar(
                    title=ft.Text("Enrollment"),
                    bgcolor=ft.Colors.SURFACE,
                    leading=ft.IconButton(ft.Icons.ARROW_BACK, on_click=back)
                ),
                EnrollmentView(self.stop_event, self.run_state_event,self.pipeline_manager, self.fps),
            ]
        )
