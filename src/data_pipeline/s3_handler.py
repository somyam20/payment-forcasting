import os
import logging
from typing import Optional
from botocore.exceptions import ClientError
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_BUCKET = os.getenv("AWS_BUCKET", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

if not AWS_BUCKET:
    logger.warning("AWS_BUCKET not set. S3 operations will fail until configured.")

def _get_client():
    kwargs = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        kwargs.update({
            "aws_access_key_id": AWS_ACCESS_KEY,
            "aws_secret_access_key": AWS_SECRET_KEY
        })
    return boto3.client("s3", **kwargs)

def upload_file(local_path: str, s3_key: str) -> str:
    """
    Uploads a local file to S3 and returns the s3 uri.
    """
    client = _get_client()
    try:
        client.upload_file(local_path, AWS_BUCKET, s3_key)
        s3_uri = f"s3://{AWS_BUCKET}/{s3_key}"
        logger.info(f"Uploaded {local_path} -> {s3_uri}")
        return s3_uri
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise

def download_file(s3_key: str, local_path: str) -> str:
    """
    Downloads an object from S3 to local_path and returns local_path.
    """
    client = _get_client()
    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        client.download_file(AWS_BUCKET, s3_key, local_path)
        logger.info(f"Downloaded s3://{AWS_BUCKET}/{s3_key} -> {local_path}")
        return local_path
    except ClientError as e:
        logger.error(f"S3 download failed: {e}")
        raise

def generate_presigned_url(s3_key: str, expires_in: int = 3600) -> Optional[str]:
    """
    Returns a presigned URL for GET. None on failure.
    """
    client = _get_client()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET, "Key": s3_key},
            ExpiresIn=expires_in
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to create presigned URL: {e}")
        return None
