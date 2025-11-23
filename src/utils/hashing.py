"""
File integrity utilities for ISO 27001 compliance.

Provides SHA256 hashing functionality for model file verification.
"""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_sha256(file_path: str, chunk_size: int = 8192) -> Optional[str]:
    """
    Calculate SHA256 hash of a file in chunks to handle large model files.

    Args:
        file_path: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)

    Returns:
        Hexadecimal string of SHA256 hash, or None if file cannot be read

    Example:
        >>> hash_val = calculate_sha256("/path/to/model.gguf")
        >>> print(f"SHA256: {hash_val}")
    """
    try:
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read file in chunks to avoid loading entire model into memory
            for byte_block in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None


def verify_hash(file_path: str, expected_hash: str) -> bool:
    """
    Verify if a file's SHA256 hash matches the expected value.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA256 hash (hexadecimal string)

    Returns:
        True if hash matches, False otherwise

    Example:
        >>> is_valid = verify_hash("/path/to/model.gguf", "abc123...")
        >>> if not is_valid:
        ...     print("File integrity check failed!")
    """
    actual_hash = calculate_sha256(file_path)

    if actual_hash is None:
        return False

    return actual_hash.lower() == expected_hash.lower()
