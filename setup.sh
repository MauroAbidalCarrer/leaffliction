#!/bin/bash

echo "Setting up project:"
pip install -qqr requirements.txt                                                               # Install other packages.
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -qq  # Install pytorch.
echo "-Installed python packages."
wget --no-verbose -qnc https://cdn.intra.42.fr/document/document/17547/leaves.zip               # Download leaves images.
echo "-Downloaded leaves images."
unzip -nq leaves.zip                                                                            # Unzip leaves.
echo "-Unzipped images."
mkdir -p images/Apple images/Grape                                                              # Add subdirectories for each fruit to match the hierarchy in the subject.
mv images/Apple_* images/Apple/                                                                 # Move all of the apple images into Apple directory.
mv images/Grape_* images/Grape/                                                                 # Move all of the grape images into Grape directory.
echo "-Reordered images hierarchy."
echo "Setup done."