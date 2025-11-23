import hashlib

def calculate_sha256(file_path, chunk_size=8192):
    """
    Calculates SHA256 hash of the model file.
    CRITICAL for ISO 27001 to prove the model wasn't tampered with.
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()