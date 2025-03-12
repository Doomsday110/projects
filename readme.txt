Project Overview:
This project processes and augments video data, extracts optical flow features, and trains a deep learning model (CNN-LSTM) to predict motion trajectories. The pipeline includes video augmentation, dataset preparation, training, evaluation, and testing.

Directory Structure:
/project_root
│── videos/            # Original video dataset
│── train_videos/      # Training set (70% of augmented videos)
│── eval_videos/       # Evaluation set (30% of augmented videos)
│── test_videos/       # Test set (Original videos)
│── augmented_videos/  # Augmented videos (mirrored & flipped)
│── model/             # Saved trained model
│── train_flow.pkl     # Optical flow data for training
│── eval_flow.pkl      # Optical flow data for evaluation
│── test_flow.pkl      # Optical flow data for testing
│── task.ipynb         # Jupyter Notebook with full implementation
│── README.docx        # This file

Dependencies:
Ensure you have the following Python libraries installed:
torch, torchvision, opencv-python, numpy, tqdm, matplotlib, scikit-learn, pickle, shutil

Install them using:
pip install torch torchvision opencv-python numpy tqdm matplotlib scikit-learn

Key Features:
1. Video Augmentation - Applies horizontal mirroring & vertical flipping.
2. Optical Flow Computation - Extracts motion information using Farneback's Optical Flow.
3. Deep Learning Model (CNN-LSTM) - Extracts spatial and temporal dependencies.
4. Training Pipeline - Uses MSE Loss with AdamW optimizer and learning rate scheduling.
5. Testing - Evaluates model performance and selects the best motion trajectory.

How to Run
1. Prepare Data - Place `.mp4` videos inside the `videos/` folder.
2. Run the script - Open `task.ipynb` and execute all cells.
3. View Results - The trained model is saved as `model/best_model.pth`.

Future Improvements
• Support for additional augmentations like rotation and brightness change.
• Implement bidirectional LSTM for better motion prediction.
• Fine-tune CNN architecture for improved feature extraction.
