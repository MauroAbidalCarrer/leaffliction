#!/bin/bash

# Source directory containing the image categories and diseases
src_dir="images"

# Destination directory to store the copied images
dest_dir="images_copy"

# Number of images to copy from each leaf directory
num_images=300

# Function to copy images
copy_images() {
    local src_leaf_dir=$1
    local dest_leaf_dir=$2
    
    # Creating the destination leaf directory
    mkdir -p "$dest_leaf_dir"
    
    # Finding and copying the first 'num_images' files from the source leaf directory
    find "$src_leaf_dir" -maxdepth 1 -type f | head -n $num_images | xargs -I {} cp {} "$dest_leaf_dir"
}

# Main script logic
for category in "$src_dir"/*; do
    if [ -d "$category" ]; then
        for disease in "$category"/*; do
            if [ -d "$disease" ]; then
                # Constructing the destination leaf directory path
                dest_leaf_dir="$dest_dir/${category##*/}/${disease##*/}"
                
                # Copying the images
                copy_images "$disease" "$dest_leaf_dir"
            fi
        done
    fi
done

echo "Copying completed!"
