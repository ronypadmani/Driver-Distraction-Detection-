Driver Distraction Detection System

A Driver Distraction Detection System identifies whether a driver is focused or distracted by analyzing dashboard camera video frames.
It studies driver posture, hand movements, and phone usage to understand driver behaviour.
This helps improve road safety by detecting risky actions early and preventing accidents.

📚 Dataset 
🚗 Synthetic Distracted Driving (SynDD1) Dataset

The SynDD1 dataset is a synthetic distracted-driving dataset available on the Mendeley Data platform.
It contains various distraction activities such as texting, drinking, eating, reaching behind, talking to passengers, and normal driving.
The dataset provides diverse, high-quality images for training and research in distracted driver detection.

📊 Dataset Overview

Column Name	Description
video_name	Name of the source video
frame_filename	Frame image name
frame_path	File path of each frame
timestamp	Time of the frame in the video
label	0 = Not Distracted, 1 = Distracted
class_id	Same as label

🛠️ Feature Extraction Summary
📸 Extracts frames from dashboard videos
👤 Detects driver pose using body keypoints
📱 Identifies objects such as phone, hands, steering wheel
🔗 Builds scene relationships (e.g., “hand holding phone”)
📏 Measures important distances (hand ↔ wheel, phone ↔ face)
🧬 Generates image embeddings from each frame
📁 Creates a combined JSON feature file for each frame

🚀 Use Cases

Driver distraction detection
Deep learning model training and evaluation
Road-safety and behavior analytics
Advanced driver assistance systems (ADAS)
Real-time driver monitoring and alert systems
Behavior classification from video frames

▶️ How to Run the App

Install dependencies:

pip install -r requirements.txt

📌 Note

The SynDD1 dataset is publicly available for research and educational purposes only.
Use of the dataset must follow the licensing terms on the platform.
