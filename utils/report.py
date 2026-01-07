from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
from pathlib import Path
import os
import cv2

# Generate a PDF report for blocked content
def generate_report(post, report_folder, posts_file="posts.json"):
    """
    Creates a professional 'Content Blocked' report with:
    - Header, red warning box
    - Detection info based on content type
    - Preview of blocked content (image/video/audio/text)
    """
    # Ensure report folder exists
    report_folder = Path(report_folder)
    report_folder.mkdir(parents=True, exist_ok=True)
    filename = f"report_{post['id']}.pdf"
    path = report_folder / filename

    # Create PDF canvas
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # ---------- HEADER ----------
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.red)
    c.drawString(72, height - 72, "🚫 BLOCKED CONTENT REPORT")

    # Post info: ID, user, type, upload time
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    c.drawString(72, height - 100, f"Report ID: {post.get('id','N/A')}")
    c.drawString(72, height - 118, f"User: {post.get('user','anonymous')}")
    c.drawString(72, height - 136, f"Type: {post.get('type','unknown').capitalize()}")
    upload_time = post.get("time")
    readable_time = datetime.fromtimestamp(upload_time).strftime("%Y-%m-%d %H:%M:%S") \
        if upload_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(72, height - 154, f"Upload Time: {readable_time}")

    # ---------- REASON ----------
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, height - 184, "Reason for Rejection:")
    c.setFont("Helvetica", 11)
    reason = str(post.get("reason","Content classified as fake or inappropriate."))
    reason = reason.split("with confidence")[0].strip()
    text_obj = c.beginText(72, height - 202)
    for line in reason.splitlines():
        text_obj.textLine(line.strip())
    c.drawText(text_obj)

    # ---------- RED BLOCKED CONTENT BOX ----------
    box_top = height - 260
    box_height = 50
    box_width = width - 144
    c.setFillColor(colors.red)
    c.roundRect(72, box_top - box_height, box_width, box_height, 10, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.white)
    c.drawCentredString(72 + box_width/2, box_top - box_height/2 - 5, "⚠ CONTENT BLOCKED!")

    # ---------- BLOCKED CONTENT DETAILS ----------
    detail_y = box_top - box_height - 30
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(72, detail_y, "Blocked Content Details & Detection Basis:")
    detail_y -= 16
    c.setFont("Helvetica", 11)

    content_type = post.get("type","unknown").lower()
    detection_points = []
    border_color = colors.black

    # Specify detection info and border color based on content type
    if content_type == "image":
        detection_points = [
            "Detected using EfficientNet-B0: Image authenticity detection",
            "Content flagged as fake based on visual features and deep learning classification"
        ]
        border_color = colors.red
    elif content_type == "video":
        detection_points = [
            "Frame-level EfficientNet: Video verification",
            "Temporal inconsistencies and manipulations detected across frames"
        ]
        border_color = colors.orange
    elif content_type == "audio":
        detection_points = [
            "Wav2Vec2: Audio deepfake detection",
            "Voice patterns analyzed to identify synthesized or manipulated audio"
        ]
        border_color = colors.blue
    elif content_type == "text":
        detection_points = [
            "DistilBERT: Text analysis and fake content classification",
            "Abusive or misleading content detected using NLP-based classifier"
        ]
        border_color = colors.grey

    # Write detection points
    for point in detection_points:
        c.drawString(80, detail_y, f"• {point}")
        detail_y -= 14

    # ---------- CONTENT PREVIEW ----------
    preview_y = detail_y - 10
    content_path = post.get("file_path")
    if content_path and os.path.exists(content_path):
        try:
            if content_type == "image":
                c.setStrokeColor(border_color)
                c.setLineWidth(3)
                c.rect(70, preview_y - 150, 4*inch, 3*inch, stroke=1, fill=0)
                c.drawImage(content_path, 72, preview_y - 150, width=4*inch, height=3*inch, preserveAspectRatio=True)
            elif content_type == "video":
                cap = cv2.VideoCapture(content_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    temp_img = "temp_frame.jpg"
                    cv2.imwrite(temp_img, frame)
                    c.setStrokeColor(border_color)
                    c.setLineWidth(3)
                    c.rect(70, preview_y - 150, 4*inch, 3*inch, stroke=1, fill=0)
                    c.drawImage(temp_img, 72, preview_y - 150, width=4*inch, height=3*inch, preserveAspectRatio=True)
                    os.remove(temp_img)
            elif content_type == "audio":
                c.setStrokeColor(border_color)
                c.setLineWidth(3)
                c.rect(70, preview_y - 80, 4*inch, 80, stroke=1, fill=0)
                c.setFont("Helvetica-Bold", 40)
                c.setFillColor(border_color)
                c.drawCentredString(72 + 2*inch, preview_y - 40, "🔊 Audio Content Blocked")
            elif content_type == "text":
                c.setStrokeColor(border_color)
                c.setLineWidth(2)
                c.rect(70, preview_y - 150, 4*inch, 3*inch, stroke=1, fill=0)
                c.setFont("Helvetica", 11)
                lines = str(post.get("content","")).splitlines()
                line_y = preview_y - 10
                for line in lines[:20]:
                    c.drawString(72, line_y, line)
                    line_y -= 14
        except Exception as e:
            c.drawString(72, preview_y, "Content preview not available.")

    # ---------- FOOTER ----------
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.black)
    c.drawString(72, 50, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(72, 35, "Automatically generated by AI-Powered Fake Content Detection System.")

    c.showPage()
    c.save()
    print(f"[✅] Blocked report generated successfully: {filename}")
    return filename
