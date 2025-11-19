- A Driver Distraction Detection System identifies whether a driver is focused or distracted by analyzing dashboard camera images.
- It understands driver behavior by detecting hand movements, phone usage, and overall body posture inside the vehicle.
- This helps improve road safety by recognizing risky situations early and reducing chances of accidents.

# 🚗 Synthetic Distracted Driving (SynDD1) Dataset

- The SynDD1 dataset is a high-quality synthetic distracted driving dataset available on the Mendeley Data platform.
- It contains multiple distraction behaviors such as texting, talking to passengers, eating, drinking, adjusting controls, and normal driving.
- The dataset is designed specifically for building reliable distracted driving detection models.
- Its synthetic nature provides clean variation in drivers, environments, and actions.

## 📊 Dataset Overview

- Column Name	Description
- video_name	Name of the source video
- frame_filename	Frame image name
- frame_path	File path of the frame
- timestamp	Time of frame in the video
- label	0 = Not Distracted, 1 = Distracted
- class_id	Same as label

## 🛠️ Feature Extraction 

- 📸 Frames are extracted from dashboard videos
- 👤 Body pose landmarks are detected
- 📱 Objects like hands, phone, and steering wheel are identified
- 🔗 Action relationships are recognized (example: “hand holding phone”)
- 📏 Distances between important points help detect distraction
- 🧬 Image embeddings are generated for each frame
- 📁 A combined JSON feature file is created per frame

## Use Cases

- Driver distraction detection
- Deep learning model training and evaluation
- Road-safety and behavior analytics
- Advanced driver assistance systems (ADAS)
- Real-time driver monitoring and alert systems
- Behavior classification from video frames

# Install dependencies
```bash
pip install -r requirements.txt
```

## 📌 Note 
- The SynDD1 dataset is publicly available for research and educational purposes only.
-  Use of the dataset must follow the licensing terms on the platform.
