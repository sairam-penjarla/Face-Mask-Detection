# Download Data from here:
   https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data


# Face Mask Detection

This project demonstrates a face mask detection model using TensorFlow and OpenCV. The model is trained to distinguish between images of people with and without face masks. Additionally, it includes a real-time detection script that uses the webcam to detect faces and predict whether they are wearing a mask.

## Project Structure

- `config.py`: Configuration file for the face mask detection model.
- `utils.py`: Utility functions for loading and preprocessing data.
- `main.py`: Script to load data, build, train, and evaluate the model.
- `detect_webcam.py`: Script to perform real-time face mask detection using the webcam.

## Setup

1. **Clone the repository**
   ```bash
   https://github.com/sairam-penjarla/Face-Mask-Detection.git
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Place your dataset in the directory specified in `config.py`. The default path is `./data`.

4. **Train the model**
   ```bash
   python main.py
   ```
   This will load the data, build the model, train it, and evaluate its performance. The trained model will be saved as `face_mask_detection_model.h5`.

5. **Run real-time detection**
   ```bash
   python detect_webcam.py
   ```
   This will start the webcam and perform real-time face mask detection.

## Requirements

The dependencies are listed in `requirements.txt`. Ensure you have Python 3.6 or later installed. 

## Notes

- The model uses the VGG16 architecture for transfer learning.
- Make sure you have a working webcam for the real-time detection script.
- The Haar Cascade file for face detection is included with OpenCV and is loaded automatically.