#!/bin/bash

# Directory containing the symbolic links
SYMLINK_DIR="/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK79/preps/C1/thumbnail"

# Old and new parts of the target path (example: /old_path/file -> /new_path/file)
OLD_TARGET_PART="../../../thumbnail_original"
NEW_TARGET_PART="../../thumbnail_original"

# Loop through each symbolic link in the directory
for link in "$SYMLINK_DIR"/*; do
  if [ -L "$link" ]; then # Check if it's a symbolic link
    # Get the current target of the symbolic link
    current_target=$(readlink "$link")

    # Check if the old target part exists in the current target
    if [[ "$current_target" == *"$OLD_TARGET_PART"* ]]; then
      # Replace the old target part with the new one
      new_target="${current_target//$OLD_TARGET_PART/$NEW_TARGET_PART}"

      # Update the symbolic link to point to the new target
      ln -sfT "$new_target" "$link"
      echo "Updated symbolic link '$link' to point to '$new_target'"
    fi
  fi
done
