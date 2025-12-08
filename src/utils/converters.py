import cv2
import base64
from io import BytesIO
from PIL import Image


def frame_to_base64(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG", quality=85)
    return base64.b64encode(buff.getvalue()).decode()
