# models/blockchain_model.py
from sqlalchemy import Column, Integer, String, Text, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import json
import os
import hashlib

# Base class for SQLAlchemy models
Base = declarative_base()

# Define the ledger table for blockchain entries
class LedgerEntry(Base):
    __tablename__ = "blockchain_ledger"  # table name in SQLite
    id = Column(Integer, primary_key=True, autoincrement=True)  # auto ID
    timestamp = Column(Integer, nullable=False)  # block creation time
    uploader_id = Column(String(128), nullable=True)  # user who uploaded content
    filename = Column(String(512), nullable=True)  # file name
    content_type = Column(String(50), nullable=True)  # type of content
    is_blocked = Column(Boolean, default=False)  # whether the content is blocked
    payload_json = Column(Text, nullable=False)  # store full payload as JSON string
    prev_hash = Column(String(128), nullable=True)  # hash of previous block
    hash = Column(String(128), nullable=False, unique=True)  # current block hash

    # Convert DB entry to dictionary for easier use
    def to_dict(self):
        d = {
            "id": self.id,
            "timestamp": self.timestamp,
            "uploader_id": self.uploader_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "is_blocked": self.is_blocked,
            "payload": json.loads(self.payload_json),  # convert JSON string to dict
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }
        return d

# Create SQLite engine
def get_engine(db_path="data/app.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)  # make folder if not exists
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    return engine

# Create blockchain ledger table in database
def create_tables(db_path="data/app.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

# Get a DB session for transactions
def get_session(db_path="data/app.db"):
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()

# Add a new ledger entry (block) to the blockchain table
def append_ledger_entry(uploader_id=None, filename=None, content_type=None, is_blocked=False, extra_payload=None, db_path="data/app.db"):
    from utils.blockchain import make_block  # function to create block with hash
    import json
    import time

    session = get_session(db_path)  # start DB session

    # Get previous block's hash
    last_entry = session.query(LedgerEntry).order_by(LedgerEntry.id.desc()).first()
    prev_hash = last_entry.hash if last_entry else ""

    # Create payload dictionary
    payload = {
        "uploader_id": uploader_id,
        "filename": filename,
        "content_type": content_type,
        "is_blocked": bool(is_blocked),
        "extra": extra_payload or {}  # extra info if provided
    }

    # Generate new block with hash linked to previous
    block = make_block(payload, prev_hash=prev_hash)

    # Create new DB entry
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

    session.add(new_entry)  # add entry to DB
    session.flush()  # ensures ID is assigned before commit

    entry_id = new_entry.id  # capture assigned ID
    session.commit()  # save to DB
    session.close()  # close session

    return entry_id, block["hash"]  # return DB ID and block hash

# Compute SHA-256 hash of a file
def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path,"rb") as f:
        # Read file in chunks and update hash
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()  # return hex string of hash
