#!/bin/bash

echo "Setting up project:"
pip install -qqr requirements.txt                                                               # Install other packages.
echo "-Installed python packages."
wget --no-verbose -qnc https://cdn.intra.42.fr/document/document/17547/leaves.zip               # Download leaves images.
echo "-Downloaded leaves images."
unzip -nq leaves.zip                                                                            # Unzip leaves.
echo "-Unzipped images."
mkdir -p images/Apple images/Grape                                                              # Add subdirectories for each fruit to match the hierarchy in the subject.
mv images/Apple_* images/Apple/                                                                 # Move all of the apple images into Apple directory.
mv images/Grape_* images/Grape/                                                                 # Move all of the grape images into Grape directory.
echo "-Reordered images hierarchy."
rm -f leaves.zip
echo "-Removed leaves zip file."
for file in images/Apple/*/*                                                                    # Rename each image to replace spaces by underscores.
do
  mv -- "$file" "${file// /_}"
done
echo "-Changed spaces to underscores in filenames."
echo "Setup done."