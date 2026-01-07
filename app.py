from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os, uuid, json, time
from pathlib import Path
import hashlib
from PIL import Image
import imagehash
import cv2

# Disable CUDA for CPU-only processing
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -------------------- Import Models & Utilities --------------------
from models.fake_image_model import predict_image, predict_folder, predict_image_from_array
from models.toxic_text_model import ToxicTextModel
from models.audio_model import classify_audio
from models.video_model import detect_video
from utils.report import generate_report
from models.blockchain_model import get_session, LedgerEntry, create_tables
from utils.blockchain import compute_file_hash, compute_text_hash, make_block

# -------------------- Paths & Folders --------------------
BASE = Path(__file__).parent
UPLOAD_FOLDER = BASE / "uploads"
REPORT_FOLDER = BASE / "reports"
DATA_FILE = BASE / "data" / "posts.json"

# Create folders if they don't exist
for d in [UPLOAD_FOLDER, REPORT_FOLDER, DATA_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# Initialize blockchain tables
create_tables(db_path="data/app.db")

# Initialize posts.json if missing
if not DATA_FILE.exists():
    DATA_FILE.write_text(json.dumps({"posts": [], "blocked": []}, indent=2))

# -------------------- Utility Functions --------------------
def save_db(db):
    """Save posts/blocked posts to JSON"""
    DATA_FILE.write_text(json.dumps(db, indent=2))

def load_db():
    """Load posts/blocked posts from JSON"""
    return json.loads(DATA_FILE.read_text())

def find_post(db, post_id):
    """Find post by ID in posts or blocked lists"""
    for list_name in ("posts", "blocked"):
        for post in db[list_name]:
            if post.get("id") == post_id:
                return post, list_name
    return None, None

@app.template_filter('datetimeformat')
def datetimeformat(value):
    """Format timestamp for templates"""
    import datetime
    return datetime.datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M:%S")

# Load toxic text model
text_model = ToxicTextModel("models/saved_harassing_model")

# -------------------- Blockchain Ledger Utilities --------------------
def is_duplicate(file_hash):
    """Check if file/text already exists in blockchain"""
    session = get_session("data/app.db")
    exists = session.query(LedgerEntry).filter(LedgerEntry.payload_json.like(f'%{file_hash}%')).first() is not None
    session.close()
    return exists

def append_ledger(uploader_id, filename, content_type, is_blocked, file_hash, extra=None):
    """Append a new block to the blockchain ledger"""
    session = get_session("data/app.db")
    last_entry = session.query(LedgerEntry).order_by(LedgerEntry.id.desc()).first()
    prev_hash = last_entry.hash if last_entry else ""
    
    payload = {"uploader_id": uploader_id, "filename": filename, "content_type": content_type,
               "is_blocked": is_blocked, "file_sha256": file_hash}
    if extra:
        payload.update(extra)
    
    block = make_block(payload, prev_hash=prev_hash)
    
    new_entry = LedgerEntry(
        timestamp=block["timestamp"],
        uploader_id=uploader_id,
        filename=filename,
        content_type=content_type,
        is_blocked=is_blocked,
        payload_json=json.dumps(payload, sort_keys=True),
        prev_hash=prev_hash,
        hash=block["hash"]
    )
    session.add(new_entry)
    session.commit()
    
    entry_id = new_entry.id
    entry_hash = block["hash"]
    session.close()
    
    return entry_id, entry_hash

# -------------------- Near-Duplicate Detection --------------------
def compute_image_phash(image_path):
    """Compute perceptual hash for an image"""
    img = Image.open(image_path)
    return str(imagehash.phash(img))

def is_image_near_duplicate(phash, threshold=5):
    """Check if an image is near-duplicate"""
    session = get_session("data/app.db")
    entries = session.query(LedgerEntry).all()
    for entry in entries:
        payload = json.loads(entry.payload_json)
        existing_phash = payload.get("file_phash")
        if existing_phash and imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(phash) <= threshold:
            session.close()
            return True
    session.close()
    return False

def compute_video_phash(video_path):
    """Compute perceptual hashes for first 30 frames of a video"""
    vidcap = cv2.VideoCapture(video_path)
    frame_hashes = []
    count = 0
    while True:
        success, frame = vidcap.read()
        if not success or count > 30:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_hashes.append(str(imagehash.phash(img)))
        count += 1
    vidcap.release()
    return frame_hashes

def is_video_near_duplicate(frame_hashes, threshold=5):
    """Check if a video is near-duplicate by comparing frame hashes"""
    session = get_session("data/app.db")
    entries = session.query(LedgerEntry).all()
    for entry in entries:
        payload = json.loads(entry.payload_json)
        existing_frames = payload.get("file_frame_hashes")
        if existing_frames:
            matches = sum(
                imagehash.hex_to_hash(f1) - imagehash.hex_to_hash(f2) <= threshold
                for f1, f2 in zip(frame_hashes, existing_frames)
            )
            if matches / len(frame_hashes) > 0.7:
                session.close()
                return True
    session.close()
    return False

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=['GET','POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username','guest')
        session['username'] = username
        session['is_admin'] = username.lower() == "admin"
        flash("Logged in as " + username, "success")
        return redirect(url_for('upload'))
    return render_template("login.html")

@app.route("/logout")
def logout():
    """Handle logout"""
    session.pop('username', None)
    session.pop('is_admin', None)
    flash("Logged out", "info")
    return redirect(url_for('index'))

# -------------------- Upload --------------------
@app.route("/upload", methods=['GET','POST'])
def upload():
    """Handle uploading content (image/video/audio/text)"""
    if request.method == 'POST':
        ctype = request.form.get('content_type')
        user = session.get('username', 'anonymous')
        file = request.files.get('file')
        text_content = request.form.get('text_content', '').strip()
        uid = str(uuid.uuid4())[:8]
        db = load_db()

        filename, path, file_hash = None, None, None

        # ----------------- File Upload Handling -----------------
        if file and file.filename:
            filename = f"{uid}_" + secure_filename(file.filename)[:100]
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            file_hash = compute_file_hash(path)

            # Duplicate file check
            if is_duplicate(file_hash):
                flash("Duplicate file detected ❌ Upload blocked.", "danger")
                append_ledger(user, filename, ctype, True, file_hash, {"reason": "Duplicate file"})
                os.remove(path)
                return redirect(url_for('blocked'))

            # Near-duplicate checks
            if ctype == 'image':
                phash = compute_image_phash(path)
                if is_image_near_duplicate(phash):
                    flash("Near-duplicate image detected ❌ Upload blocked.", "danger")
                    append_ledger(user, filename, ctype, True, file_hash, {"file_phash": phash, "reason": "Near-duplicate image"})
                    os.remove(path)
                    return redirect(url_for('blocked'))
            elif ctype == 'video':
                frame_hashes = compute_video_phash(path)
                if is_video_near_duplicate(frame_hashes):
                    flash("Near-duplicate video detected ❌ Upload blocked.", "danger")
                    append_ledger(user, filename, ctype, True, file_hash, {"file_frame_hashes": frame_hashes, "reason": "Near-duplicate video"})
                    os.remove(path)
                    return redirect(url_for('blocked'))

        # ----------------- Text Handling -----------------
        elif text_content:
            file_hash = compute_text_hash(text_content)
            if is_duplicate(file_hash):
                flash("Duplicate text detected ❌ Upload blocked.", "danger")
                append_ledger(user, None, ctype, True, file_hash, {"reason": "Duplicate text"})
                return redirect(url_for('blocked'))

        # ----------------- Fake Detection -----------------
        try:
            if ctype == 'image' and path:
                result = predict_image(path)
                is_fake, reason = result.get("fake"), result.get("reason")
            elif ctype == 'video' and path:
                result = detect_video(path)
                is_fake = result.lower() == "fake"
                reason = f"Video detected as {result}"
            elif ctype == 'audio' and path:
                result = classify_audio(path)
                is_fake = result["label"].lower() == "fake"
                reason = f"Detected as {result['label']} ({result['confidence']:.2f}%)"
            else:
                text_check = text_model.predict([text_content])
                is_fake = text_check[0].lower() == "fake"
                reason = f"Text classified as {text_check[0]}"
        except Exception as e:
            flash(f"Error during detection: {e}", "danger")
            return redirect(url_for('upload'))

        # Additional payload info
        extra_payload = {}
        if ctype == 'image' and path:
            extra_payload["file_phash"] = compute_image_phash(path)
        elif ctype == 'video' and path:
            extra_payload["file_frame_hashes"] = compute_video_phash(path)

        # Append entry to blockchain ledger
        append_ledger(user, filename, ctype, is_fake, file_hash, {**extra_payload, "reason": reason})

        # ----------------- Save Post -----------------
        post = {
            "id": uid,
            "user": user,
            "type": ctype,
            "filename": filename,
            "text": text_content,
            "time": int(time.time()),
            "reason": reason if is_fake else "",
            "likes": 0,
            "comments": []
        }

        # Block or allow post
        if is_fake:
            db['blocked'].append(post)
            save_db(db)
            generate_report(post, REPORT_FOLDER)  # create PDF report
            flash("Content blocked ❌ Report generated.", "danger")
            return redirect(url_for('blocked'))
        else:
            db['posts'].append(post)
            save_db(db)
            flash("Content uploaded successfully ✅", "success")
            return redirect(url_for('feed'))

    return render_template("upload.html")

# -------------------- Like & Comment APIs --------------------
@app.route('/like/<post_id>', methods=['POST'])
def like_post(post_id):
    db = load_db()
    post, _ = find_post(db, post_id)
    if not post:
        return jsonify({"success": False})
    post['likes'] = post.get('likes', 0) + 1
    save_db(db)
    return jsonify({"success": True, "likes": post['likes']})

@app.route('/comment/<post_id>', methods=['POST'])
def comment_post(post_id):
    db = load_db()
    post, _ = find_post(db, post_id)
    if not post:
        return jsonify({"success": False})
    try:
        data = request.get_json()
        text = data.get('comment', '').strip()
    except:
        return jsonify({"success": False})
    if not text:
        return jsonify({"success": False})
    comment = {"user": session.get('username', 'Anonymous'), "text": text, "time": int(time.time())}
    post.setdefault('comments', []).append(comment)
    save_db(db)
    return jsonify({"success": True, "comments": post['comments']})

@app.route('/api/post/<post_id>')
def api_post(post_id):
    """Get post likes & comments via API"""
    db = load_db()
    post, _ = find_post(db, post_id)
    if not post:
        return jsonify({"likes":0, "comments":[]})
    return jsonify({"likes": post.get('likes',0), "comments": post.get('comments', [])})

# -------------------- Feed & Blocked Pages --------------------
@app.route("/feed")
def feed():
    """Show all allowed posts in feed"""
    db = load_db()
    posts = list(reversed(db.get('posts', [])))
    return render_template("feed.html", posts=posts)

@app.route("/blocked")
def blocked():
    """Show all blocked posts"""
    db = load_db()
    blocked = list(reversed(db.get('blocked', [])))
    return render_template("blocked.html", blocked=blocked)

# -------------------- Delete Post --------------------
@app.route("/delete/<post_id>", methods=["POST","DELETE"])
def delete_post(post_id):
    """Delete a post and its file from uploads"""
    db = load_db()
    deleted = False
    for list_name in ["posts","blocked"]:
        for post in list(db[list_name]):
            if post["id"] == post_id:
                if post.get("filename"):
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], post["filename"])
                    if os.path.exists(file_path):
                        try: os.remove(file_path)
                        except Exception as e: print(f"Error deleting file: {e}")
                db[list_name].remove(post)
                deleted = True
                break
        if deleted: break
    if deleted:
        save_db(db)
        return jsonify({"success": True,"message":"Post deleted successfully"}),200
    else:
        return jsonify({"success": False,"message":"Post not found"}),404

# -------------------- Reports & Uploads --------------------
@app.route("/reports/<path:filename>")
def report_file(filename):
    """Serve report PDFs"""
    return send_from_directory(str(REPORT_FOLDER), filename, as_attachment=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(str(UPLOAD_FOLDER), filename)

# -------------------- Ledger --------------------
@app.route("/ledger")
def ledger_view():
    """Show blockchain ledger entries"""
    session_db = get_session("data/app.db")
    entries = session_db.query(LedgerEntry).order_by(LedgerEntry.id.desc()).all()
    entries = [e.to_dict() for e in entries]
    session_db.close()
    return render_template("ledger.html", entries=entries)

@app.route('/delete_entry/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    """Delete ledger entry and corresponding file"""
    session = get_session("data/app.db")
    entry = session.query(LedgerEntry).filter(LedgerEntry.id==entry_id).first()
    if not entry:
        session.close()
        return jsonify({'success': False, 'message': 'Entry not found.'})
    
    filename = None
    try:
        payload = json.loads(entry.payload_json)
        filename = payload.get("filename")
    except: pass

    session.delete(entry)
    session.commit()
    session.close()

    # Delete uploaded file if exists
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({'success': True})

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
