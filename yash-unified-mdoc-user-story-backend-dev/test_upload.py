#!/usr/bin/env python3
"""
Utility script to upload a local video (or any binary file) to S3 and print the public URL.

Example:
    python test_upload.py --file data/temp/09882fb1-d9b7-4c9f-9d01-97ecb5a59bde.mp4 --folder uploads/videos
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.utils.s3_utility import S3Utility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a file to S3 and return its URL.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the local video/file to upload.",
    )
    parser.add_argument(
        "--folder",
        default="uploads/videos",
        help="Destination folder inside the S3 bucket (default: uploads/videos).",
    )
    parser.add_argument(
        "--prefix",
        default="mdoc-upload",
        help="Optional prefix prepended to the uploaded filename.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_path = Path(args.file).expanduser().resolve()

    if not source_path.exists() or not source_path.is_file():
        print(f"❌ File not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    file_bytes = source_path.read_bytes()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest_filename = f"{args.prefix}-{timestamp}-{source_path.name}"

    s3 = S3Utility()
    if not s3.bucket_name:
        print(
            "❌ S3_BUCKET_NAME environment variable is not set. Please configure it before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        url = s3.upload_file(file_bytes, dest_filename, args.folder)
    except Exception as exc:
        print(f"❌ Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Uploaded successfully: {url}")


if __name__ == "__main__":
    main()
    # python test_upload.py --file data/temp/your-video.mp4 --folder uploads/videos --prefix myrun
