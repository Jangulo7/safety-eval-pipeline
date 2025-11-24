"""Unit tests for utils/hashing.py - File integrity utilities."""

import os

import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.utils.hashing import calculate_sha256, verify_hash  # noqa: E402


class TestCalculateSHA256:
    """Tests for SHA256 hash calculation."""

    def test_hash_small_file(self, temp_dir):
        """Test hashing a small file."""
        test_file = os.path.join(temp_dir, "test.txt")
        content = b"Hello, World!"

        with open(test_file, "wb") as f:
            f.write(content)

        result = calculate_sha256(test_file)

        # SHA256 of "Hello, World!"
        expected = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        assert result == expected

    def test_hash_large_file(self, temp_dir):
        """Test hashing a large file (chunked reading)."""
        test_file = os.path.join(temp_dir, "large.bin")

        # Create 10MB file
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        result = calculate_sha256(test_file)

        # Should return a valid hex string
        assert result is not None
        assert len(result) == 64  # SHA256 hex is 64 characters
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_empty_file(self, temp_dir):
        """Test hashing an empty file."""
        test_file = os.path.join(temp_dir, "empty.txt")

        with open(test_file, "wb"):
            pass  # Create empty file

        result = calculate_sha256(test_file)

        # SHA256 of empty file
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert result == expected

    def test_hash_nonexistent_file(self):
        """Test hashing a file that doesn't exist."""
        result = calculate_sha256("/nonexistent/file.txt")
        assert result is None

    def test_hash_with_custom_chunk_size(self, temp_dir):
        """Test hashing with custom chunk size."""
        test_file = os.path.join(temp_dir, "test.bin")

        with open(test_file, "wb") as f:
            f.write(b"Test content")

        result = calculate_sha256(test_file, chunk_size=4)

        # Should produce same result regardless of chunk size
        assert result is not None


class TestVerifyHash:
    """Tests for hash verification."""

    def test_verify_correct_hash(self, temp_dir):
        """Test verification with correct hash."""
        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "wb") as f:
            f.write(b"Test content")

        # Calculate expected hash
        expected_hash = calculate_sha256(test_file)

        # Verify
        assert verify_hash(test_file, expected_hash) is True

    def test_verify_incorrect_hash(self, temp_dir):
        """Test verification with incorrect hash."""
        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "wb") as f:
            f.write(b"Test content")

        wrong_hash = "0" * 64

        assert verify_hash(test_file, wrong_hash) is False

    def test_verify_case_insensitive(self, temp_dir):
        """Test that hash verification is case-insensitive."""
        test_file = os.path.join(temp_dir, "test.txt")

        with open(test_file, "wb") as f:
            f.write(b"Test")

        hash_lower = calculate_sha256(test_file)
        hash_upper = hash_lower.upper()

        assert verify_hash(test_file, hash_upper) is True

    def test_verify_nonexistent_file(self):
        """Test verification of nonexistent file."""
        result = verify_hash("/nonexistent/file.txt", "abc123")
        assert result is False
