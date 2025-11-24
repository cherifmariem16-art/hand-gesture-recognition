# hand-gesture-recognition
Real-time hand gesture recognition using OpenCV and MediaPipe.

## Description
This Python project performs real-time hand gesture recognition using a webcam. It leverages Mediapipe for hand-tracking and OpenCV for video capture and display. The system can detect both single-hand and two-hand gestures, including:
* Good / Bad (thumb up/down)
* Stop (open hand)
* Point (pointing gesture)
* Run (closed hand)
* Peace

## Dependencies

This project requires the following Python libraries:

| Library           | Version | Purpose |
|------------------|---------|---------|
| `numpy`           | 1.25.0 | For numerical operations and array handling |
| `opencv-python`   | 4.8.0  | For image and video processing |
| `mediapipe`       | 0.12.0 | For hand-tracking and pose detection |

### Installation

Install the dependencies using pip:

```bash
pip install numpy==1.25.0 opencv-python==4.8.0 mediapipe==0.12.0


