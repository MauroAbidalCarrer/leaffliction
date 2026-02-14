#!/usr/bin/env bash

set -e

ARCHIVE_NAME="submission.zip"

# Check inputs exist
if [ ! -d "dataset" ]; then
    echo "Error: dataset directory not found."
    exit 1
fi

if [ ! -f "model.pt" ]; then
    echo "Error: model.pt not found."
    exit 1
fi

# Create zip archive
zip -r "$ARCHIVE_NAME" dataset model.pt

# Generate SHA1 signature
sha1sum "$ARCHIVE_NAME" > signature.txt

echo "Archive created: $ARCHIVE_NAME"
echo "SHA1 signature written to: signature.txt"
