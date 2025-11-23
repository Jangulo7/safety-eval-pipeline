"""
Shared AWS clients with connection pooling.

This module provides singleton instances of AWS clients to avoid
creating new connections for every operation, which improves performance
and reduces connection overhead.
"""
import logging
from typing import Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# Module-level singletons
_s3_client = None
_s3_resource = None


def get_s3_client(region_name: Optional[str] = None):
    """
    Get a shared S3 client instance with connection pooling.

    This function returns a singleton boto3 S3 client that is reused
    across the application, reducing connection overhead.

    Args:
        region_name: AWS region (default: uses AWS_DEFAULT_REGION env var)

    Returns:
        boto3.client: Configured S3 client

    Example:
        >>> from aws_clients import get_s3_client
        >>> s3 = get_s3_client()
        >>> s3.list_buckets()
    """
    global _s3_client

    if _s3_client is None:
        # Configure with connection pooling and retries
        config = Config(
            region_name=region_name,
            retries={
                "max_attempts": 3,
                "mode": "adaptive",  # Adaptive retry mode
            },
            max_pool_connections=50,  # Connection pool size
            tcp_keepalive=True,
        )

        _s3_client = boto3.client("s3", config=config)
        logger.info("Initialized shared S3 client")

    return _s3_client


def get_s3_resource(region_name: Optional[str] = None):
    """
    Get a shared S3 resource instance with connection pooling.

    Similar to get_s3_client() but returns a resource interface
    which provides a higher-level object-oriented API.

    Args:
        region_name: AWS region (default: uses AWS_DEFAULT_REGION env var)

    Returns:
        boto3.resource: Configured S3 resource

    Example:
        >>> from aws_clients import get_s3_resource
        >>> s3 = get_s3_resource()
        >>> bucket = s3.Bucket('my-bucket')
    """
    global _s3_resource

    if _s3_resource is None:
        config = Config(
            region_name=region_name,
            retries={
                "max_attempts": 3,
                "mode": "adaptive",
            },
            max_pool_connections=50,
            tcp_keepalive=True,
        )

        _s3_resource = boto3.resource("s3", config=config)
        logger.info("Initialized shared S3 resource")

    return _s3_resource


def reset_clients():
    """
    Reset all client singletons.

    Useful for testing or when you need to force recreation of clients
    with different configurations.
    """
    global _s3_client, _s3_resource
    _s3_client = None
    _s3_resource = None
    logger.info("Reset all AWS clients")
