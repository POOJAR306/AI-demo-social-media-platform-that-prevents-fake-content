import hashlib
import json
from datetime import datetime
from pathlib import Path

# -----------------------------
# Compute SHA256 hash of a file
# -----------------------------
def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file in chunks to handle large files."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------
# Compute SHA256 hash of a text string
# -----------------------------
def compute_text_hash(text: str) -> str:
    """Compute SHA256 hash of any text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -----------------------------
# Create a blockchain-style block
# -----------------------------
def make_block(payload: dict, prev_hash: str = "") -> dict:
    """
    Create a block dictionary with readable timestamp, payload, previous hash, and current hash.
    payload: dictionary with content info (uploader_id, filename, content_type, is_blocked, etc.)
    prev_hash: hash of the previous block in the chain
    """
    block = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # readable time
        "payload": payload,                                         # info about the content
        "prev_hash": prev_hash,                                     # link to previous block
    }

    # Convert block dict to a JSON string with consistent order for hashing
    block_string = json.dumps(block, sort_keys=True, separators=(',', ':'))

    # Compute SHA256 hash of the block string
    block_hash = hashlib.sha256(block_string.encode("utf-8")).hexdigest()

    # Add hash to block
    block["hash"] = block_hash

    return block  # return full block dictionary

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    payload = {
        "uploader_id": 1,
        "filename": "example.jpg",
        "content_type": "image",
        "is_blocked": False
    }

    block = make_block(payload)
    print(json.dumps(block, indent=4))
