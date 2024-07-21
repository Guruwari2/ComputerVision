# Face Morphing Project

## Overview

This project implements a face morphing technique that allows users to morph between two facial images. The main script, `main.py`, handles the entire morphing process. Additionally, users can create GIFs of the morphing process, manually add matching points, and specify morphing values using command-line options.

## Features

- **Automatic Morphing**: Run `main.py` to compute the morphing between two images.
- **GIF Creation**: Use the `--gif` option to create a GIF of the morphing process.
- **Manual Matching Points**: Use the `--manualpoints` option to manually add matching points between the images.
- **Specify Morphing Value**: Use the `--wfrac` option to specify the morphing value.

## Prerequisites

Ensure you have the following packages installed:
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- Imageio

You can install the required packages using:
```sh
pip install numpy opencv-python matplotlib imageio
