"""
1. Scene Graphs (RelTR-like)
2. Pose Estimation (OpenPose)
3. Object Detection (YOLO)
4. Distance Features
5. Image Embeddings (VGG16/ResNet)
"""

import cv2
import numpy as np
import pandas as pd
import json
import torch
import torchvision.transforms as transforms
from pathlib import Path
import mediapipe as mp
from ultralytics import YOLO
import torchvision.models as models
from PIL import Image
import math
from tqdm import tqdm
import argparse

class FeatureExtractor:
    def __init__(self, features_dir="features"):
        """
        Initialize feature extraction pipeline
        
        Args:
            features_dir: Directory to save extracted features
        """
        self.features_dir = Path(features_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create feature subdirectories
        self.scene_graphs_dir = self.features_dir / "scene_graphs"
        self.poses_dir = self.features_dir / "poses"
        self.objects_dir = self.features_dir / "objects"
        self.distances_dir = self.features_dir / "distances"
        self.embeddings_dir = self.features_dir / "embeddings"
        self.combined_dir = self.features_dir / "combined"
        
        for dir_path in [self.scene_graphs_dir, self.poses_dir, self.objects_dir, 
                        self.distances_dir, self.embeddings_dir, self.combined_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all feature extraction models"""
        print("Initializing feature extraction models...")
        
        # 1. MediaPipe for pose estimation (alternative to OpenPose)
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # 2. YOLO for object detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
            print(" YOLO model loaded")
        except Exception as e:
            print(f" YOLO model loading failed: {e}")
            self.yolo_model = None
        
        # 3. VGG16 for image embeddings
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = self.vgg16.classifier[:-1]  
        self.vgg16.eval()
        self.vgg16.to(self.device)
        
        # Image preprocessing for VGG16
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(" All models initialized successfully")
    
    def extract_scene_graph(self, image_path, frame_name):
        """
        Extract scene graph features (simplified version)
        In a full implementation, this would use RelTR or similar
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Simplified scene graph: detect basic objects and relationships
            # This is a placeholder - in production, use RelTR or similar
            scene_graph = {
                "objects": [
                    {"id": 0, "class": "person", "bbox": [0, 0, 100, 100], "confidence": 0.9},
                    {"id": 1, "class": "steering_wheel", "bbox": [150, 200, 50, 50], "confidence": 0.8},
                    {"id": 2, "class": "phone", "bbox": [300, 100, 30, 60], "confidence": 0.7}
                ],
                "relationships": [
                    {"subject": 0, "predicate": "holding", "object": 2, "confidence": 0.6},
                    {"subject": 0, "predicate": "near", "object": 1, "confidence": 0.8}
                ],
                "frame_name": frame_name
            }
            
            # Save scene graph
            output_path = self.scene_graphs_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(scene_graph, f, indent=2)
            
            return scene_graph
            
        except Exception as e:
            print(f"Error extracting scene graph for {frame_name}: {e}")
            return None
    
    def extract_pose_features(self, image_path, frame_name):
        """
        Extract pose estimation features using MediaPipe
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Body pose detection
            pose_results = self.pose_detector.process(image_rgb)
            
            # Hand detection
            hands_results = self.hands_detector.process(image_rgb)
            
            pose_features = {
                "frame_name": frame_name,
                "body_keypoints": [],
                "hand_keypoints": [],
                "pose_detected": False,
                "hands_detected": False
            }
            
            # Extract body keypoints
            if pose_results.pose_landmarks:
                pose_features["pose_detected"] = True
                for landmark in pose_results.pose_landmarks.landmark:
                    pose_features["body_keypoints"].append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
            
            # Extract hand keypoints
            if hands_results.multi_hand_landmarks:
                pose_features["hands_detected"] = True
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    hand_points = []
                    for landmark in hand_landmarks.landmark:
                        hand_points.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })
                    pose_features["hand_keypoints"].append(hand_points)
            
            # Save pose features
            output_path = self.poses_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(pose_features, f, indent=2)
            
            return pose_features
            
        except Exception as e:
            print(f"Error extracting pose features for {frame_name}: {e}")
            return None
    
    def extract_object_features(self, image_path, frame_name):
        """
        Extract object detection features using YOLO
        """
        try:
            if self.yolo_model is None:
                return None
            
            # Run YOLO detection
            results = self.yolo_model(str(image_path))
            
            object_features = {
                "frame_name": frame_name,
                "objects": [],
                "total_objects": 0
            }
            
            # Extract detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        object_features["objects"].append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                        })
            
            object_features["total_objects"] = len(object_features["objects"])
            
            # Save object features
            output_path = self.objects_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(object_features, f, indent=2)
            
            return object_features
            
        except Exception as e:
            print(f"Error extracting object features for {frame_name}: {e}")
            return None
    
    def calculate_distance_features(self, pose_features, object_features, frame_name):
        """
        Calculate distance features between key points
        """
        try:
            distance_features = {
                "frame_name": frame_name,
                "distances": {},
                "spatial_relationships": []
            }
            
            # Calculate distances if we have pose and object data
            if pose_features and object_features:
                # Find hands from pose
                hands_positions = []
                if pose_features.get("hands_detected") and pose_features["hand_keypoints"]:
                    for hand in pose_features["hand_keypoints"]:
                        if hand:  # Check if hand has keypoints
                            # Get wrist position (first keypoint)
                            wrist = hand[0]
                            hands_positions.append([wrist["x"], wrist["y"]])
                
                # Find relevant objects
                steering_wheel = None
                phone = None
                
                for obj in object_features["objects"]:
                    if "wheel" in obj["class_name"].lower() or obj["class_name"] == "car":
                        steering_wheel = obj["center"]
                    elif "phone" in obj["class_name"].lower() or "cell phone" in obj["class_name"].lower():
                        phone = obj["center"]
                
                # Calculate key distances
                if hands_positions and steering_wheel:
                    for i, hand_pos in enumerate(hands_positions):
                        dist = math.sqrt((hand_pos[0] - steering_wheel[0])**2 + 
                                       (hand_pos[1] - steering_wheel[1])**2)
                        distance_features["distances"][f"hand_{i}_to_steering_wheel"] = dist
                
                if hands_positions and phone:
                    for i, hand_pos in enumerate(hands_positions):
                        dist = math.sqrt((hand_pos[0] - phone[0])**2 + 
                                       (hand_pos[1] - phone[1])**2)
                        distance_features["distances"][f"hand_{i}_to_phone"] = dist
                
                # Face to phone distance (if face detected in pose)
                if pose_features.get("pose_detected") and pose_features["body_keypoints"] and phone:
                    # Nose is typically keypoint 0 in pose landmarks
                    if len(pose_features["body_keypoints"]) > 0:
                        nose = pose_features["body_keypoints"][0]
                        face_to_phone_dist = math.sqrt((nose["x"] - phone[0])**2 + 
                                                     (nose["y"] - phone[1])**2)
                        distance_features["distances"]["face_to_phone"] = face_to_phone_dist
            
            # Save distance features
            output_path = self.distances_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(distance_features, f, indent=2)
            
            return distance_features
            
        except Exception as e:
            print(f"Error calculating distance features for {frame_name}: {e}")
            return None
    
    def extract_image_embeddings(self, image_path, frame_name):
        """
        Extract image embeddings using VGG16
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                embeddings = self.vgg16(image_tensor)
                embeddings = embeddings.cpu().numpy().flatten()
            
            embedding_features = {
                "frame_name": frame_name,
                "embedding_size": len(embeddings),
                "embedding": embeddings.tolist()
            }
            
            # Save embeddings
            output_path = self.embeddings_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(embedding_features, f, indent=2)
            
            return embedding_features
            
        except Exception as e:
            print(f"Error extracting embeddings for {frame_name}: {e}")
            return None
    
    def combine_features(self, scene_graph, pose_features, object_features, 
                        distance_features, embedding_features, frame_name):
        """
        Combine all features into a single JSON
        """
        try:
            combined_features = {
                "frame_name": frame_name,
                "timestamp": None,  # Will be filled from CSV
                "features": {
                    "scene_graph": scene_graph,
                    "pose": pose_features,
                    "objects": object_features,
                    "distances": distance_features,
                    "embeddings": {
                        "size": embedding_features["embedding_size"] if embedding_features else 0,
                        "vector": embedding_features["embedding"] if embedding_features else []
                    }
                },
                "feature_summary": {
                    "has_scene_graph": scene_graph is not None,
                    "has_pose": pose_features is not None and pose_features.get("pose_detected", False),
                    "has_hands": pose_features is not None and pose_features.get("hands_detected", False),
                    "num_objects": object_features["total_objects"] if object_features else 0,
                    "num_distances": len(distance_features["distances"]) if distance_features else 0,
                    "embedding_size": embedding_features["embedding_size"] if embedding_features else 0
                }
            }
            
            # Save combined features
            output_path = self.combined_dir / f"{frame_name}.json"
            with open(output_path, 'w') as f:
                json.dump(combined_features, f, indent=2)
            
            return combined_features
            
        except Exception as e:
            print(f"Error combining features for {frame_name}: {e}")
            return None
    
    def process_frame(self, image_path, frame_name):
        """
        Process a single frame through the entire feature extraction pipeline
        """
        print(f"Processing frame: {frame_name}")
        
        # Extract all feature types
        scene_graph = self.extract_scene_graph(image_path, frame_name)
        pose_features = self.extract_pose_features(image_path, frame_name)
        object_features = self.extract_object_features(image_path, frame_name)
        distance_features = self.calculate_distance_features(pose_features, object_features, frame_name)
        embedding_features = self.extract_image_embeddings(image_path, frame_name)
        
        # Combine all features
        combined_features = self.combine_features(
            scene_graph, pose_features, object_features, 
            distance_features, embedding_features, frame_name
        )
        
        return combined_features
    
    def process_dataset(self, frames_csv_path):
        """
        Process entire dataset from frames_labels.csv
        """
        print("Loading frames dataset...")
        df = pd.read_csv(frames_csv_path)
        
        print(f"Processing {len(df)} frames...")
        
        processed_count = 0
        failed_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            frame_path = Path(row['frame_path'])
            frame_name = Path(row['frame_filename']).stem
            
            if frame_path.exists():
                try:
                    combined_features = self.process_frame(frame_path, frame_name)
                    if combined_features:
                        processed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error processing {frame_name}: {e}")
                    failed_count += 1
            else:
                print(f"Frame not found: {frame_path}")
                failed_count += 1
            
            # Progress update every 100 frames
            if (idx + 1) % 100 == 0:
                print(f"Processed: {processed_count}, Failed: {failed_count}")
        
        print(f"\n=== Feature Extraction Complete ===")
        print(f" Successfully processed: {processed_count} frames")
        print(f" Failed: {failed_count} frames")
        print(f" Features saved in: {self.features_dir}")
        
        return processed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Extract features for driver distraction detection')
    parser.add_argument('--frames_csv', default='dataset/frames_labels.csv', 
                       help='Path to frames labels CSV')
    parser.add_argument('--features_dir', default='features', 
                       help='Directory to save features')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Process only first N frames (for testing)')
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(features_dir=args.features_dir)
    
    # Load and optionally sample dataset
    df = pd.read_csv(args.frames_csv)
    if args.sample_size:
        df = df.head(args.sample_size)
        print(f"Processing sample of {len(df)} frames")
    
    # Process dataset
    processed, failed = extractor.process_dataset(args.frames_csv)
    
    print("Feature extraction pipeline completed!")

if __name__ == "__main__":
    main()
