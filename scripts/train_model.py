"""
Model Training Script for Driver Distraction Detection
Loads combined features and trains a CombinerModel (MLP/LSTM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

class DriverDistractionDataset(Dataset):
    """Dataset class for driver distraction detection"""
    
    def __init__(self, features_dir, frames_csv, max_samples=None):
        """
        Initialize dataset
        
        Args:
            features_dir: Directory containing combined feature JSONs
            frames_csv: CSV file with frame labels
            max_samples: Maximum number of samples to load (for testing)
        """
        self.features_dir = Path(features_dir)
        self.combined_dir = self.features_dir / "combined"
        
        # Load frame labels
        self.df = pd.read_csv(frames_csv)
        if max_samples:
            self.df = self.df.head(max_samples)
        
        # Filter only frames that have feature files
        self.valid_samples = []
        for idx, row in self.df.iterrows():
            frame_name = Path(row['frame_filename']).stem
            feature_file = self.combined_dir / f"{frame_name}.json"
            if feature_file.exists():
                self.valid_samples.append({
                    'frame_name': frame_name,
                    'feature_file': feature_file,
                    'label': row['label'],
                    'video_name': row['video_name']
                })
        
        print(f"Dataset initialized with {len(self.valid_samples)} valid samples")
        
        # Calculate feature dimensions by loading first sample
        if self.valid_samples:
            self._calculate_feature_dims()
    
    def _calculate_feature_dims(self):
        """Calculate feature dimensions from first sample"""
        first_sample = self.valid_samples[0]
        features = self._load_features(first_sample['feature_file'])
        
        self.scene_graph_dim = len(features['scene_graph_vector'])
        self.pose_dim = len(features['pose_vector'])
        self.object_dim = len(features['object_vector'])
        self.distance_dim = len(features['distance_vector'])
        self.embedding_dim = len(features['embedding_vector'])
        self.total_dim = (self.scene_graph_dim + self.pose_dim + 
                         self.object_dim + self.distance_dim + self.embedding_dim)
        
        print(f"Feature dimensions:")
        print(f"  Scene graph: {self.scene_graph_dim}")
        print(f"  Pose: {self.pose_dim}")
        print(f"  Objects: {self.object_dim}")
        print(f"  Distances: {self.distance_dim}")
        print(f"  Embeddings: {self.embedding_dim}")
        print(f"  Total: {self.total_dim}")
    
    def _load_features(self, feature_file):
        """Load and process features from JSON file"""
        with open(feature_file, 'r') as f:
            data = json.load(f)
        
        features = data.get('features', {})
        
        # Process scene graph features
        scene_graph = features.get('scene_graph', {})
        scene_graph_vector = self._process_scene_graph(scene_graph)
        
        # Process pose features
        pose = features.get('pose', {})
        pose_vector = self._process_pose(pose)
        
        # Process object features
        objects = features.get('objects', {})
        object_vector = self._process_objects(objects)
        
        # Process distance features
        distances = features.get('distances', {})
        distance_vector = self._process_distances(distances)
        
        # Process embedding features
        embeddings = features.get('embeddings', {})
        embedding_vector = embeddings.get('vector', [])
        
        # Ensure consistent dimensions
        if not embedding_vector:
            embedding_vector = [0.0] * 4096  # VGG16 default size
        
        return {
            'scene_graph_vector': scene_graph_vector,
            'pose_vector': pose_vector,
            'object_vector': object_vector,
            'distance_vector': distance_vector,
            'embedding_vector': embedding_vector
        }
    
    def _process_scene_graph(self, scene_graph):
        """Convert scene graph to fixed-size vector"""
        # Simplified scene graph representation
        # In production, use proper graph neural networks
        vector = [0.0] * 20  # Fixed size vector
        
        if scene_graph and 'objects' in scene_graph:
            objects = scene_graph['objects']
            # Count object types
            person_count = sum(1 for obj in objects if obj.get('class') == 'person')
            steering_wheel_count = sum(1 for obj in objects if obj.get('class') == 'steering_wheel')
            phone_count = sum(1 for obj in objects if obj.get('class') == 'phone')
            
            vector[0] = min(person_count, 5.0)  # Cap at 5
            vector[1] = min(steering_wheel_count, 2.0)
            vector[2] = min(phone_count, 2.0)
            vector[3] = len(objects)  # Total objects
            
            # Average confidence
            if objects:
                avg_confidence = sum(obj.get('confidence', 0) for obj in objects) / len(objects)
                vector[4] = avg_confidence
        
        return vector
    
    def _process_pose(self, pose):
        """Convert pose keypoints to fixed-size vector"""
        vector = [0.0] * 100  # Fixed size for pose features
        
        if pose and pose.get('pose_detected'):
            body_keypoints = pose.get('body_keypoints', [])
            if body_keypoints:
                # Flatten first 33 keypoints (x, y, z, visibility)
                for i, kp in enumerate(body_keypoints[:25]):  # Use first 25 keypoints
                    if i * 4 + 3 < len(vector):
                        vector[i * 4] = kp.get('x', 0)
                        vector[i * 4 + 1] = kp.get('y', 0)
                        vector[i * 4 + 2] = kp.get('z', 0)
                        vector[i * 4 + 3] = kp.get('visibility', 0)
        
        # Add hand detection flag
        if pose and pose.get('hands_detected'):
            vector[99] = 1.0
        
        return vector
    
    def _process_objects(self, objects):
        """Convert YOLO objects to fixed-size vector"""
        vector = [0.0] * 50  # Fixed size for object features
        
        if objects and 'objects' in objects:
            obj_list = objects['objects']
            
            # Count specific object types relevant to driving
            person_count = sum(1 for obj in obj_list if obj.get('class_name') == 'person')
            car_count = sum(1 for obj in obj_list if obj.get('class_name') == 'car')
            phone_count = sum(1 for obj in obj_list if 'phone' in obj.get('class_name', '').lower())
            
            vector[0] = min(person_count, 5.0)
            vector[1] = min(car_count, 2.0)
            vector[2] = min(phone_count, 2.0)
            vector[3] = len(obj_list)  # Total objects
            
            # Average confidence
            if obj_list:
                avg_confidence = sum(obj.get('confidence', 0) for obj in obj_list) / len(obj_list)
                vector[4] = avg_confidence
                
                # Bounding box features (normalized)
                for i, obj in enumerate(obj_list[:10]):  # First 10 objects
                    bbox = obj.get('bbox', [0, 0, 0, 0])
                    if len(bbox) >= 4:
                        base_idx = 5 + i * 4
                        if base_idx + 3 < len(vector):
                            vector[base_idx] = bbox[0] / 1920.0  # Normalize by image width
                            vector[base_idx + 1] = bbox[1] / 1080.0  # Normalize by image height
                            vector[base_idx + 2] = (bbox[2] - bbox[0]) / 1920.0  # Width
                            vector[base_idx + 3] = (bbox[3] - bbox[1]) / 1080.0  # Height
        
        return vector
    
    def _process_distances(self, distances):
        """Convert distance features to fixed-size vector"""
        vector = [0.0] * 10  # Fixed size for distance features
        
        if distances and 'distances' in distances:
            dist_dict = distances['distances']
            
            # Key distances for distraction detection
            if 'hand_0_to_steering_wheel' in dist_dict:
                vector[0] = min(dist_dict['hand_0_to_steering_wheel'], 2.0)
            if 'hand_1_to_steering_wheel' in dist_dict:
                vector[1] = min(dist_dict['hand_1_to_steering_wheel'], 2.0)
            if 'hand_0_to_phone' in dist_dict:
                vector[2] = min(dist_dict['hand_0_to_phone'], 2.0)
            if 'hand_1_to_phone' in dist_dict:
                vector[3] = min(dist_dict['hand_1_to_phone'], 2.0)
            if 'face_to_phone' in dist_dict:
                vector[4] = min(dist_dict['face_to_phone'], 2.0)
        
        return vector
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load features
        features = self._load_features(sample['feature_file'])
        
        # Combine all features
        combined_vector = (
            features['scene_graph_vector'] +
            features['pose_vector'] +
            features['object_vector'] +
            features['distance_vector'] +
            features['embedding_vector']
        )
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(combined_vector)
        label_tensor = torch.LongTensor([sample['label']])
        
        return feature_tensor, label_tensor.squeeze()

class CombinerModel(nn.Module):
    """Neural network model for driver distraction detection"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        """
        Initialize the combiner model
        
        Args:
            input_dim: Total input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(CombinerModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    """Training manager for the driver distraction model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc="Training"):
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validation"):
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0
        best_model_state = None
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {best_val_acc:.2f}%")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return best_val_acc, val_preds, val_labels
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history saved to {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train driver distraction detection model')
    parser.add_argument('--features_dir', default='features', help='Features directory')
    parser.add_argument('--frames_csv', default='dataset/frames_labels.csv', help='Frames labels CSV')
    parser.add_argument('--model_path', default='models/driver_distraction_model.pth', help='Model save path')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for testing')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create models directory
    Path(args.model_path).parent.mkdir(exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = DriverDistractionDataset(args.features_dir, args.frames_csv, args.max_samples)
    
    if len(dataset) == 0:
        print(" No valid samples found! Make sure feature extraction is completed.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CombinerModel(input_dim=dataset.total_dim)
    trainer = ModelTrainer(model, device)
    
    print(f"Model created with input dimension: {dataset.total_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    best_acc, val_preds, val_labels = trainer.train(
        train_loader, val_loader, 
        num_epochs=args.epochs, 
        lr=args.lr
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': dataset.total_dim,
        'feature_dims': {
            'scene_graph_dim': dataset.scene_graph_dim,
            'pose_dim': dataset.pose_dim,
            'object_dim': dataset.object_dim,
            'distance_dim': dataset.distance_dim,
            'embedding_dim': dataset.embedding_dim
        },
        'best_accuracy': best_acc
    }, args.model_path)
    
    print(f"\n Model saved to {args.model_path}")
    print(f" Best validation accuracy: {best_acc:.2f}%")
    
    # Print classification report
    print("\n Classification Report:")
    print(classification_report(val_labels, val_preds, 
                              target_names=['Not Distracted', 'Distracted']))
    
    # Plot training history
    trainer.plot_training_history('models/training_history.png')
    
    print("\n Model training completed successfully!")

if __name__ == "__main__":
    main()
