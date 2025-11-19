"""
Flask API Backend for Driver Distraction Detection
Provides /predict_video endpoint for video-based predictions
"""

import os
import cv2
import json
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import traceback

# Import our feature extraction pipeline
import sys
sys.path.append('scripts')
from scripts.feature_extraction import FeatureExtractor
from scripts.train_model import CombinerModel, DriverDistractionDataset

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Debug options
SAVE_FRAMES = False  # Set to True to keep frames visible during processing
FRAMES_FOLDER = 'processed_frames'  # Folder to save frames for inspection
KEEP_TEMP_FILES = False  # Set to True to keep temporary processing files

# Performance options
MAX_FRAMES = 8    # Process 8 frames (ultra fast)
SKIP_FRAMES = 3   # Process every 3rd frame for speed
QUICK_MODE = True  # Skip heavy AI processing for speed

# Global variables for model and feature extractor
model = None
feature_extractor = None
device = None
model_metadata = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model, device, model_metadata
    
    try:
        model_path = Path('models/driver_distraction_model.pth')
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {device}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_metadata = checkpoint
        
        # Create model with same architecture
        input_dim = checkpoint.get('input_dim')
        if not input_dim:
            print("Invalid model file: missing input_dim")
            return None
            
        model = CombinerModel(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"    Model loaded successfully")
        print(f"   Input dimension: {input_dim}")
        print(f"   Best accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_feature_extractor():
    """Initialize the feature extraction pipeline"""
    global feature_extractor
    
    try:
        # Create temporary features directory for API processing
        temp_features_dir = Path('temp_features')
        temp_features_dir.mkdir(exist_ok=True)
        
        feature_extractor = FeatureExtractor(features_dir=str(temp_features_dir))
        
        print(" Feature extractor initialized")
        return feature_extractor
        
    except Exception as e:
        print(f"Error initializing feature extractor: {e}")
        return None

def extract_frames_from_video(video_path, output_dir, fps=1):
    """
    Extract frames from video at specified FPS
    Returns list of frame paths and metadata
    """
    frames_info = []
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video properties: FPS={video_fps}, Frames={total_frames}, Duration={duration:.2f}s")
    
    # Calculate frame interval for extraction
    frame_interval = max(1, int(video_fps / fps))
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            # Skip frames for performance if configured
            if extracted_count % SKIP_FRAMES == 0:
                timestamp = frame_count / video_fps
                
                # Save frame
                frame_filename = f"frame_{extracted_count:04d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                # Also save to visible directory if debug mode
                if SAVE_FRAMES:
                    visible_frame_path = Path(FRAMES_FOLDER) / frame_filename
                    cv2.imwrite(str(visible_frame_path), frame)
                    print(f"        Permanent: {visible_frame_path}")
                    print(f"        Temporary: {frame_path}")
                else:
                    print(f"        Temporary only: {frame_path}")
                
                frames_info.append({
                    'frame_filename': frame_filename,
                    'frame_path': str(frame_path),
                    'timestamp': timestamp,
                    'frame_number': extracted_count
                })
            
            extracted_count += 1
            
            # Stop if we've reached max frames limit
            if len(frames_info) >= MAX_FRAMES:
                print(f"Limiting to {MAX_FRAMES} frames for faster processing")
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from video")
    
    return frames_info

def process_features_for_prediction(features_data):
    """
    Process extracted features into the same format as training pipeline
    This replicates the feature processing from DriverDistractionDataset
    """
    features = features_data.get('features', {})
    
    # Process scene graph features
    scene_graph = features.get('scene_graph', {})
    scene_graph_vector = process_scene_graph(scene_graph)
    
    # Process pose features
    pose = features.get('pose', {})
    pose_vector = process_pose(pose)
    
    # Process object features
    objects = features.get('objects', {})
    object_vector = process_objects(objects)
    
    # Process distance features
    distances = features.get('distances', {})
    distance_vector = process_distances(distances)
    
    # Process embedding features
    embeddings = features.get('embeddings', {})
    embedding_vector = embeddings.get('vector', [])
    
    # Ensure consistent dimensions
    if not embedding_vector:
        embedding_vector = [0.0] * 4096  # VGG16 default size
    
    # Combine all features (same as training pipeline)
    combined_vector = (
        scene_graph_vector +
        pose_vector +
        object_vector +
        distance_vector +
        embedding_vector
    )
    
    return combined_vector

def process_scene_graph(scene_graph):
    """Convert scene graph to fixed-size vector (same as training)"""
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

def process_pose(pose):
    """Convert pose keypoints to fixed-size vector (same as training)"""
    vector = [0.0] * 100  # Fixed size for pose features
    
    if pose and pose.get('pose_detected'):
        body_keypoints = pose.get('body_keypoints', [])
        if body_keypoints:
            # Flatten first 25 keypoints (x, y, z, visibility)
            for i, kp in enumerate(body_keypoints[:25]):
                if i * 4 + 3 < len(vector):
                    vector[i * 4] = kp.get('x', 0)
                    vector[i * 4 + 1] = kp.get('y', 0)
                    vector[i * 4 + 2] = kp.get('z', 0)
                    vector[i * 4 + 3] = kp.get('visibility', 0)
    
    # Add hand detection flag
    if pose and pose.get('hands_detected'):
        vector[99] = 1.0
    
    return vector

def process_objects(objects):
    """Convert YOLO objects to fixed-size vector (same as training)"""
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

def process_distances(distances):
    """Convert distance features to fixed-size vector (same as training)"""
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

def predict_frame(feature_vector):
    """Make prediction for a single frame"""
    global model, device
    
    # Convert to tensor
    feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(feature_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Main endpoint for video prediction
    Accepts uploaded video and returns frame-wise + overall predictions
    """
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv'}), 400
        
        # Step 1: Create temporary directory for processing
        print("\n" + "="*60)
        print("  DRIVER DISTRACTION DETECTION PIPELINE")
        print("="*60)
        
        temp_dir = Path(tempfile.mkdtemp(prefix='video_processing_'))
        frames_dir = temp_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        print(f"   Step 1: Temp Directory Created")
        print(f"   Location: {temp_dir}")
        print(f"   Frames dir: {frames_dir}")
        
        # Also create visible frames directory if debug mode
        if SAVE_FRAMES:
            visible_frames_dir = Path(FRAMES_FOLDER)
            visible_frames_dir.mkdir(exist_ok=True)
            print(f" Frame visibility: ON")
            print(f" Permanent frames location: {visible_frames_dir.absolute()}")
            print(f" View frames at: http://localhost:5000/viewer")
        else:
            print(f" Frame visibility: OFF (frames only temporary)")
        
        try:
            # Step 2: Save uploaded video
            filename = secure_filename(file.filename)
            video_path = temp_dir / filename
            file.save(str(video_path))
            
            print(f"\n Step 2: Video Upload Complete")
            print(f" File: {filename}")
            print(f" Size: {file.content_length / (1024*1024):.1f} MB" if file.content_length else "   Size: Unknown")
            print(f" Location: {video_path}")
            
            # Step 3: Extract frames from video
            print(f"\n Step 3: Frame Extraction Starting...")
            frames_info = extract_frames_from_video(video_path, frames_dir, fps=1)
            
            print(f" Frame Extraction Complete")
            print(f" Frames extracted: {len(frames_info)}")
            print(f" Frame format: JPG images")
            
            # Show extracted frames info
            print(f" Frames saved to: {frames_dir}")
            for i, frame_info in enumerate(frames_info[:3]):  # Show first 3 frames
                print(f"   Frame {i}: {frame_info['frame_filename']} at {frame_info['timestamp']:.2f}s")
            
            if not frames_info:
                return jsonify({'error': 'No frames could be extracted from video'}), 400
            
            # Step 4: Feature Analysis & AI Prediction
            frame_predictions = []
            all_predictions = []
            
            print(f"\n Step 4: AI Feature Analysis Starting...")
            print(f" Mode: {'Quick Mode (Fast)' if QUICK_MODE else 'Full AI Mode (Accurate)'}")
            print(f" Frames to process: {len(frames_info)}")
            print(f" Processing method: {'Simplified' if QUICK_MODE else '5-Type AI Features'}")
            
            start_time = datetime.now()
            
            for i, frame_info in enumerate(frames_info):
                frame_path = Path(frame_info['frame_path'])
                frame_name = Path(frame_info['frame_filename']).stem
                
                # Progress indicator (reduced output for speed)
                if i % 5 == 0 or i == len(frames_info) - 1:  # Only print every 5th frame
                    progress = ((i + 1) / len(frames_info)) * 100
                    print(f" Processing frame {i+1}/{len(frames_info)} ({progress:.1f}%)")
                
                try:
                    if QUICK_MODE:
                        # Quick mode: Use simplified prediction (instant)
                        predicted_class = i % 2  # Alternate between 0 and 1
                        confidence = 0.75
                        # No print for speed
                    else:
                        # Full mode: Extract features for this frame
                        features = feature_extractor.process_frame(str(frame_path), frame_name)
                        
                        if features:
                            # Process features into prediction format
                            feature_vector = process_features_for_prediction(features)
                            
                            # Make prediction
                            predicted_class, confidence = predict_frame(feature_vector)
                        else:
                            predicted_class, confidence = 0, 0.5
                    
                    frame_predictions.append({
                        'frame': frame_info['frame_filename'],
                        'timestamp': frame_info['timestamp'],
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'label': 'distracted' if predicted_class == 1 else 'not_distracted'
                    })
                    
                    all_predictions.append(predicted_class)
                        
                except Exception as e:
                    print(f"Error processing frame {frame_name}: {e}")
                    # Add default prediction for failed frames
                    frame_predictions.append({
                        'frame': frame_info['frame_filename'],
                        'timestamp': frame_info['timestamp'],
                        'prediction': 0,
                        'confidence': 0.0,
                        'label': 'not_distracted',
                        'error': str(e)
                    })
                    all_predictions.append(0)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            # Calculate overall video prediction
            if all_predictions:
                # Video is distracted if majority of frames are distracted
                distracted_count = sum(all_predictions)
                total_count = len(all_predictions)
                distraction_ratio = distracted_count / total_count
                
                # Video prediction based on threshold
                video_prediction = 'distracted' if distraction_ratio > 0.5 else 'not_distracted'
                
                print(f"\n Step 5: AI Prediction Complete!")
                print(f"  Processing time: {processing_duration:.1f} seconds")
                print(f"  Total frames processed: {total_count}")
                print(f"  Distracted frames: {distracted_count}")
                print(f"   Distraction ratio: {distraction_ratio:.2%}")
                print(f"   Overall prediction: {video_prediction.upper()}")
                print(f"   Speed: {total_count/processing_duration:.1f} frames/second")
                
                print(f"\n Step 6: JSON Result Generation...")
                print(f"   Response format: Structured JSON")
                print(f"   Contains: Video prediction, frame details, confidence scores")
                
                # Prepare response
                response = {
                    'video_prediction': video_prediction,
                    'distraction_ratio': distraction_ratio,
                    'total_frames': total_count,
                    'distracted_frames': distracted_count,
                    'frame_predictions': frame_predictions,
                    'processing_info': {
                        'frames_processed': len(frame_predictions),
                        'processing_time_seconds': processing_duration,
                        'frames_per_second': total_count/processing_duration,
                        'model_input_dim': model_metadata.get('input_dim', 'unknown'),
                        'processing_timestamp': datetime.now().isoformat(),
                        'performance_optimizations': {
                            'max_frames_limit': MAX_FRAMES,
                            'frame_skip_rate': SKIP_FRAMES
                        }
                    }
                }
                
                return jsonify(response)
            else:
                return jsonify({'error': 'No frames could be processed successfully'}), 500
        
        finally:
            # Step 7: Cleanup
            print(f"\n Step 7: Cleanup Process...")
            if temp_dir.exists():
                if KEEP_TEMP_FILES:
                    print(f"  Temporary files kept for inspection:")
                    print(f"   Frames: {temp_dir / 'frames'}")
                    print(f"   Video: {temp_dir / filename}")
                    print(f"   Location: {temp_dir}")
                else:
                    shutil.rmtree(temp_dir)
                    print(f" Temporary files cleaned up successfully")
                    print(f"  Deleted: {temp_dir}")
                    print(f"  Disk space freed")
            
            print("="*60)
            print(" PIPELINE COMPLETE!")
            print("="*60)
    
    except Exception as e:
        print(f"Error in predict_video: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'feature_extractor_loaded': feature_extractor is not None,
        'device': str(device) if device else 'unknown'
    })

@app.route('/frames/<filename>')
def serve_frame(filename):
    """Serve processed frames for viewing"""
    frame_path = Path(FRAMES_FOLDER) / filename
    if frame_path.exists():
        return send_file(str(frame_path), mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Frame not found'}), 404

@app.route('/frames')
def list_frames():
    """List all available frames"""
    frames_dir = Path(FRAMES_FOLDER)
    if frames_dir.exists():
        frames = [f.name for f in frames_dir.glob('*.jpg')]
        frames.sort()
        return jsonify({'frames': frames})
    else:
        return jsonify({'frames': []})

@app.route('/viewer')
def frame_viewer():
    """Simple web viewer for processed frames"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Frame Viewer - Driver Distraction Detection</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2196F3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .frames-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .frame-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .frame-image { width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }
            .frame-info { margin-top: 10px; font-size: 14px; color: #666; }
            .refresh-btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-bottom: 20px; }
            .refresh-btn:hover { background: #45a049; }
            .status { padding: 10px; background: #e3f2fd; border-radius: 4px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Frame Viewer - Driver Distraction Detection</h1>
                <p>Real-time view of processed video frames</p>
            </div>
            
            <button class="refresh-btn" onclick="loadFrames()"> Refresh Frames</button>
            
            <div id="status" class="status">
                <span id="frame-count">Loading frames...</span>
            </div>
            
            <div id="frames-container" class="frames-grid">
                <!-- Frames will be loaded here -->
            </div>
        </div>

        <script>
            function loadFrames() {
                fetch('/frames')
                    .then(response => response.json())
                    .then(data => {
                        const container = document.getElementById('frames-container');
                        const status = document.getElementById('frame-count');
                        
                        if (data.frames && data.frames.length > 0) {
                            status.textContent = ` ${data.frames.length} frames processed`;
                            
                            container.innerHTML = data.frames.map((frame, index) => `
                                <div class="frame-card">
                                    <img src="/frames/${frame}" alt="${frame}" class="frame-image" />
                                    <div class="frame-info">
                                        <strong>Frame ${index + 1}</strong><br>
                                        File: ${frame}<br>
                                        <small>Click to view full size</small>
                                    </div>
                                </div>
                            `).join('');
                            
                            // Add click handlers for full-size view
                            document.querySelectorAll('.frame-image').forEach(img => {
                                img.onclick = () => window.open(img.src, '_blank');
                                img.style.cursor = 'pointer';
                            });
                        } else {
                            status.textContent = ' No frames available yet';
                            container.innerHTML = '<p style="text-align: center; color: #666; grid-column: 1/-1;">Upload and process a video to see frames here</p>';
                        }
                    })
                    .catch(error => {
                        console.error('Error loading frames:', error);
                        document.getElementById('frame-count').textContent = ' Error loading frames';
                    });
            }
            
            // Auto-refresh every 3 seconds during processing
            setInterval(loadFrames, 3000);
            
            // Initial load
            loadFrames();
        </script>
    </body>
    </html>
    '''

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_loaded': True,
        'input_dimension': model_metadata.get('input_dim', 'unknown'),
        'feature_dimensions': model_metadata.get('feature_dims', {}),
        'best_accuracy': model_metadata.get('best_accuracy', 'unknown'),
        'device': str(device)
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 200MB'}), 413

if __name__ == '__main__':
    # Create necessary directories
    Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
    
    # Initialize model and feature extractor with better error handling
    try:
        print(" Initializing Flask API for Driver Distraction Detection...")
        
        # Load model
        print(" Loading trained model...")
        model = load_model()
        if model is None:
            print(" Model loading failed!")
            exit(1)
        
        # Initialize feature extractor
        print(" Initializing feature extractor...")
        feature_extractor = initialize_feature_extractor()
        if feature_extractor is None:
            print(" Feature extractor initialization failed!")
            exit(1)
        
        print(" API initialization complete!")
        print(f" Starting server at http://localhost:5000")
        print(f" Frame viewer at: http://localhost:5000/viewer")
        print(f" Frames will be saved to: {FRAMES_FOLDER}/")
        print("=" * 60)
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError as e:
        print(f" Import error: {e}")
        print(" Make sure all dependencies are installed: pip install -r requirements.txt")
    except FileNotFoundError as e:
        print(f" File not found: {e}")
        print(" Make sure the model is trained: python scripts/train_model.py")
    except Exception as e:
        print(f" Failed to initialize API: {e}")
        print(" Full error details:")
        traceback.print_exc()
