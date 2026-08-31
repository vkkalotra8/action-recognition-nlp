"""Render each slide of the PPTX to a PNG image for visual inspection."""

import os
from pathlib import Path
import win32com.client

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PPTX_PATH = PROJECT_ROOT / "reports" / "presentation" / "Action_Recognition_Final_Presentation.pptx"
IMAGE_DIR = PROJECT_ROOT / "reports" / "presentation" / "slide_images"

def export_slides():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ppt_app = win32com.client.Dispatch("PowerPoint.Application")
    presentation = None
    try:
        presentation = ppt_app.Presentations.Open(str(PPTX_PATH), WithWindow=False)
        for i, slide in enumerate(presentation.Slides):
            img_path = IMAGE_DIR / f"slide_{i+1:02d}.png"
            slide.Export(str(img_path), "PNG", 1920, 1080)
            print(f"Exported Slide {i+1} -> {img_path}")
    finally:
        if presentation:
            presentation.Close()
        ppt_app.Quit()

if __name__ == "__main__":
    export_slides()
