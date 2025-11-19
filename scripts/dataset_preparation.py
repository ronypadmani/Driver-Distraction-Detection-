"""
Dataset Preparation Script for Driver Distraction Detection
Extracts frames from videos and creates frame-level labels CSV
"""

import cv2
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import argparse

class DatasetPreparator:
    def __init__(self, videos_dir, annotations_dir, output_dir, fps=1):
        """
        Initialize dataset preparator
        
        Args:
            videos_dir: Directory containing MP4 videos
            annotations_dir: Directory containing CSV annotation files
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract (default: 1 frame per second)
        """
        self.videos_dir = Path(videos_dir)
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Store frame labels for CSV generation
        self.frame_labels = []
    
    def time_to_seconds(self, time_str):
        """Convert time string (HH:MM:SS) to seconds"""
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        except:
            return 0
    
    def load_annotations(self, csv_file):
        """Load and parse annotation CSV file"""
        df = pd.read_csv(csv_file)
        annotations = []
        
        for _, row in df.iterrows():
            start_time = self.time_to_seconds(row['Start Time'])
            end_time = self.time_to_seconds(row['End Time'])
            
            # Convert activity type to binary label
            # Distracted = 1, Not Distracted = 0
            label = 1 if row['Activity Type'].lower() == 'distracted' else 0
            
            annotations.append({
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'class_id': row['Label/Class ID']
            })
        
        return annotations
    
    def get_frame_label(self, timestamp, annotations):
        """Get label for a frame at given timestamp"""
        for ann in annotations:
            if ann['start_time'] <= timestamp <= ann['end_time']:
                return ann['label'], ann['class_id']
        
        # Default to not distracted if no annotation found
        return 0, 0
    
    def extract_frames_from_video(self, video_path, video_name, annotations):
        """Extract frames from a single video"""
        print(f"Processing video: {video_name}")
        
        # Create video-specific output directory
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        print(f"Video FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate frame interval for extraction
        frame_interval = int(video_fps / self.fps)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_count / video_fps
                
                # Get label for this timestamp
                label, class_id = self.get_frame_label(timestamp, annotations)
                
                # Save frame
                frame_filename = f"frame_{extracted_count:04d}.jpg"
                frame_path = video_output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                # Store frame info for CSV
                self.frame_labels.append({
                    'video_name': video_name,
                    'frame_filename': frame_filename,
                    'frame_path': str(frame_path),
                    'timestamp': timestamp,
                    'label': label,
                    'class_id': class_id
                })
                
                extracted_count += 1
                
                if extracted_count % 100 == 0:
                    print(f"Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames from {video_name}")
        return extracted_count
    
    def process_all_videos(self):
        """Process all videos in the videos directory"""
        print("Starting dataset preparation...")
        
        # Get all MP4 files
        video_files = list(self.videos_dir.glob("*.MP4")) + list(self.videos_dir.glob("*.mp4"))
        
        total_frames = 0
        
        for video_file in video_files:
            # Extract video name without extension
            video_name = video_file.stem
            
            # Find corresponding annotation file
            # Extract user ID from video filename
            user_id = None
            if "User_id_" in video_name:
                user_id = video_name.split("User_id_")[1].split("_")[0]
            
            if user_id:
                annotation_file = self.annotations_dir / f"user_id_{user_id}.csv"
                
                if annotation_file.exists():
                    # Load annotations
                    annotations = self.load_annotations(annotation_file)
                    
                    # Extract frames
                    frame_count = self.extract_frames_from_video(
                        video_file, video_name, annotations
                    )
                    total_frames += frame_count
                else:
                    print(f"Warning: No annotation file found for {video_name}")
            else:
                print(f"Warning: Could not extract user ID from {video_name}")
        
        print(f"Total frames extracted: {total_frames}")
        return total_frames
    
    def generate_frames_labels_csv(self):
        """Generate CSV file mapping frames to labels"""
        if not self.frame_labels:
            print("No frame labels to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.frame_labels)
        
        # Save to CSV
        csv_path = self.output_dir / "frames_labels.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Saved frame labels to {csv_path}")
        print(f"Total labeled frames: {len(df)}")
        
        # Print label distribution
        label_counts = df['label'].value_counts()
        print(f"Label distribution:")
        print(f"Not Distracted (0): {label_counts.get(0, 0)}")
        print(f"Distracted (1): {label_counts.get(1, 0)}")
        
        return csv_path

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for driver distraction detection')
    parser.add_argument('--videos_dir', default='Videos', help='Directory containing videos')
    parser.add_argument('--annotations_dir', default='Annotations', help='Directory containing annotations')
    parser.add_argument('--output_dir', default='dataset', help='Output directory for frames')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DatasetPreparator(
        videos_dir=args.videos_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        fps=args.fps
    )
    
    # Process all videos
    preparator.process_all_videos()
    
    # Generate labels CSV
    preparator.generate_frames_labels_csv()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()
