# AI-demo-social-media-platform-that-prevents-fake-content
Demo Social Media Platform (Fake Content Detection)

This is a demo social media platform built with Flask that demonstrates content upload and fake content detection for images, videos, audio, and text.

The system uses pretrained machine learning models for inference on CPU. For demonstration or if models are unavailable, it uses safe heuristics to classify content.

Features

Landing page, Login, and Upload page

Feed for approved content and Blocked content section

Fake content detection using pretrained models (CPU):

Images/Videos: EfficientNet / Torchvision models

Audio: Wav2Vec2 for classification

Text: DistilBERT for toxic/harassing/fake content detection

PDF report generation for blocked content, saved to /reports/

Uploaded files stored in /uploads/

Blockchain-style ledger for traceability of all uploads

System Requirements

Python 3.10+ recommended

No GPU required; CPU-only inference

Internet connection needed once to download pretrained models

Installation & Run

Clone the repository

git clone <repo_url>
cd demo_social_media_demo


Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / Mac


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Open in browser
http://127.0.0.1:5000
