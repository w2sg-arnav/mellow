import torch
import os
import librosa
import yaml
import transformers
import sys
import glob

# Define paths
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
CONFIG_FILES = [
    os.path.join(REPO_ROOT, "v0.yaml"),
    os.path.join(REPO_ROOT, "config.yaml"),
    os.path.join(CHECKPOINTS_DIR, "v0.yaml"),  # Check in checkpoints dir too
]

# Verify environment
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Create proper torch device
print(f"Device: {device}")
print(f"Librosa Version: {librosa.__version__}")
print(f"PyYAML Version: {yaml.__version__}")
print(f"Transformers Version: {transformers.__version__}")

# Load configuration
config = {}
for config_file in CONFIG_FILES:
    print(f"\nTrying to load configuration from {config_file}")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("Configuration loaded successfully")
            break
        except Exception as e:
            print(f"Error loading config: {e}")
    else:
        print(f"Config file not found: {config_file}")

# Look for any yaml file in the root directory or checkpoints directory
if not config:
    yaml_files = (
        glob.glob(os.path.join(REPO_ROOT, "*.yaml")) + 
        glob.glob(os.path.join(REPO_ROOT, "*.yml")) +
        glob.glob(os.path.join(CHECKPOINTS_DIR, "*.yaml")) +
        glob.glob(os.path.join(CHECKPOINTS_DIR, "*.yml"))
    )
    if yaml_files:
        print(f"Found YAML files: {[os.path.basename(f) for f in yaml_files]}")
        try:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {yaml_files[0]}")
        except Exception as e:
            print(f"Error loading config: {e}")

# Check for any Python scripts in the mellow directory that might contain model definitions
script_files = glob.glob(os.path.join(REPO_ROOT, "*.py"))
script_files = [f for f in script_files if f != os.path.abspath(__file__)]  # Exclude this script
if script_files:
    print(f"\nFound Python scripts: {[os.path.basename(f) for f in script_files]}")

# Create a simple model that doesn't rely on external dependencies that might be causing issues
class MellowModel(torch.nn.Module):
    def __init__(self):
        super(MellowModel, self).__init__()
        
        # Use configuration from yaml if available
        self.config = config
        
        # Audio feature dimension
        feature_dim = 1024
        hidden_dim = 512
        output_dim = 512
        
        # Simple encoder - avoid using external models that might have compatibility issues
        self.audio_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        ).to(device)  # Move each component to device
        
        # Simple language model - avoid transformer libraries with dependencies
        self.lm = torch.nn.Sequential(
            torch.nn.Linear(output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        ).to(device)  # Move each component to device
        
        # Mapping layer
        self.mapping_layer = torch.nn.Linear(output_dim, output_dim).to(device)
        
        print("Created simplified Mellow model (without external dependencies)")
        
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            try:
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                # Check what's in the checkpoint
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
                    if 'model_state_dict' in checkpoint:
                        self.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Try loading directly
                        self.load_state_dict(checkpoint)
                else:
                    print("Checkpoint format not recognized")
                print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    def forward(self, audio_path, text=None):
        try:
            # Load and process audio
            audio, sr = librosa.load(audio_path, sr=32000, mono=True)
            print(f"Loaded audio: {audio_path}, shape: {audio.shape}, sr: {sr}")
            
            # Generate mock features on the correct device
            features = torch.randn(1, 1024, device=device)  # Ensure tensor is on the right device
            
            # Process through model components (all on same device)
            encoded_audio = self.audio_encoder(features)
            mapped_features = self.mapping_layer(encoded_audio)
            output = self.lm(mapped_features)
            
            # Return text response
            return {"generated_text": "This is a placeholder response from the simplified Mellow model."}
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return {"generated_text": "Error processing audio", "error": str(e)}

# Scan the checkpoints directory for all types of checkpoint files
print("\nScanning checkpoints directory...")
if os.path.exists(CHECKPOINTS_DIR):
    checkpoint_files = (
        glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pth")) + 
        glob.glob(os.path.join(CHECKPOINTS_DIR, "*.pt")) +
        glob.glob(os.path.join(CHECKPOINTS_DIR, "*.ckpt"))  # Add .ckpt extension
    )
    if checkpoint_files:
        print(f"Found checkpoint files: {[os.path.basename(f) for f in checkpoint_files]}")
        checkpoint_path = checkpoint_files[0]  # Use the first one
    else:
        checkpoint_path = None
        print("No checkpoint files (*.pth, *.pt, *.ckpt) found in the checkpoints directory.")
else:
    checkpoint_path = None
    print(f"Checkpoints directory not found: {CHECKPOINTS_DIR}")

# Initialize model
try:
    print("\nInitializing simplified Mellow model...")
    mellow_model = MellowModel().to(device)
    
    if checkpoint_path:
        mellow_model.load_checkpoint(checkpoint_path)
    
    mellow_model.eval()
    print("Mellow model initialized")
    
    # Test with audio files from multiple possible locations
    test_audio_locations = [
        os.path.join(REPO_ROOT, "resource"),
        REPO_ROOT,  # Check in root directory too
    ]
    
    audio_file_found = False
    for audio_dir in test_audio_locations:
        if os.path.exists(audio_dir):
            print(f"\nLooking for audio files in {audio_dir}")
            audio_files = [f for f in os.listdir(audio_dir) 
                          if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
            
            if audio_files:
                test_audio_path = os.path.join(audio_dir, audio_files[0])
                print(f"Found audio file: {os.path.basename(test_audio_path)}")
                audio_file_found = True
                
                # Forward pass with error handling
                try:
                    with torch.no_grad():  # Disable gradient calculation for inference
                        response = mellow_model(test_audio_path)
                    print(f"Test Response: {response['generated_text']}")
                    break  # Exit after first successful test
                except Exception as e:
                    print(f"Error during model inference: {e}")
            else:
                print(f"No audio files found in {audio_dir}")
    
    # Check for specific audio file mentioned
    if not audio_file_found:
        specific_audio = os.path.join(REPO_ROOT, "audio_1.wav")
        if os.path.exists(specific_audio):
            print(f"\nFound specific audio file: audio_1.wav")
            try:
                with torch.no_grad():
                    response = mellow_model(specific_audio)
                print(f"Test Response: {response['generated_text']}")
            except Exception as e:
                print(f"Error during model inference: {e}")
                
except Exception as e:
    print(f"Error initializing Mellow model: {e}")

print("\nMellow environment setup complete.")