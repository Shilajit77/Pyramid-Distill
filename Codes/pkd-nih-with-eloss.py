"""
Multi-Teacher Knowledge Distillation for Medical Image Classification
Ensemble Training Script with Resume Functionality

This script implements a multi-teacher knowledge distillation framework
for training lightweight student models on the NIH chest X-ray dataset.
"""

import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration Constants
LABEL_MAPPING = {
    'Infiltration': 0, 'Atelectasis': 1, 'Effusion': 2, 'Nodule': 3,
    'Pneumothorax': 4, 'Mass': 5, 'Consolidation': 6, 'Pleural_Thickening': 7,
    'Cardiomegaly': 8, 'Emphysema': 9, 'Fibrosis': 10, 'Edema': 11,
    'Pneumonia': 12, 'Hernia': 13
}

NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    """Configuration class to store all hyperparameters and paths."""
    
    def __init__(self):
        # Data paths
        self.image_directory = '/csehome/m22cs062/NIH_Dataset'
        self.csv_file = 'nih_single_label.csv'
        
        # Model paths
        self.teacher_models = {
            't1': 'models/teacher1-7(res50_tune).pth',
            't2': 'models/teacher8-14(res50).pth', 
            't3': 'models/teacher1-14(res101_tune).pth'
        }
        
        # Output directories
        self.output_dir = 'ensemble-nih'
        self.plots_dir = 'plots'
        self.checkpoint_dir = 'checkpoints'
        
        # Training parameters
        self.batch_size = 16
        self.num_epochs = 500
        self.temperature = 20.0
        self.seed = 42
        
        # Data split ratios
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        # Optimizer parameters
        self.learning_rates = {
            's1': 0.06, 's2': 0.006, 's3': 0.06, 's4': 0.006
        }
        self.weight_decays = {
            's1': 5e-4, 's2': 5e-4, 's3': 5e-4, 's4': 25e-4
        }
        self.momentum = 0.9
        self.patience = 3
        
        # Loss weights
        self.ensemble_weights = {'s1': 0.2, 's2': 0.6, 's3': 0.2, 's4': 0.6}
        self.distillation_weights = {'specialist': 0.6, 'general': 0.4}


class NIHDataset(Dataset):
    """Custom Dataset class for NIH chest X-ray dataset."""
    
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, 
                 label_mapping: Dict[str, int], transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.label_mapping = label_mapping
        self.transform = transform
        self._create_file_index()
    
    def _create_file_index(self):
        """Create an index of all image files for faster lookup."""
        self.file_index = {}
        for dp, dn, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_index[f] = os.path.join(dp, f)
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.dataframe.iloc[idx, 0]
        img_path = self.file_index.get(img_name)
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in {self.root_dir}")
        
        image = Image.open(img_path).convert('RGB')
        label_str = self.dataframe.iloc[idx, 1]
        label_num = self.label_mapping.get(label_str, -1)
        
        if label_num == -1:
            raise ValueError(f"Unknown label: {label_str}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_num


class KnowledgeDistillationTrainer:
    """Main trainer class for multi-teacher knowledge distillation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_directories()
        self.setup_data()
        self.setup_models()
        self.setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_losses = [float('inf')] * 4
        self.best_models = [None] * 4
        self.train_losses = [[] for _ in range(4)]
        self.val_losses = [[] for _ in range(4)]
        self.kl_losses = []
    
    def setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config.output_dir, self.config.plots_dir, 
                         self.config.checkpoint_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_data(self):
        """Setup data loaders."""
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load dataset
        df = pd.read_csv(self.config.csv_file)
        dataset = NIHDataset(df, self.config.image_directory, 
                           LABEL_MAPPING, self.transform)
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(self.config.train_ratio * total_size)
        val_size = int(self.config.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(self.config.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    def setup_models(self):
        """Setup teacher and student models."""
        # Load teacher models
        self.teachers = {}
        
        # Teacher 1: Classes 0-6 (ResNet50)
        t1 = models.resnet50(pretrained=True)
        t1.fc = nn.Linear(t1.fc.in_features, 7)
        t1.load_state_dict(torch.load(self.config.teacher_models['t1'], 
                                    map_location=DEVICE))
        t1.eval()
        self.teachers['t1'] = t1.to(DEVICE)
        
        # Teacher 2: Classes 7-13 (ResNet50)
        t2 = models.resnet50(pretrained=True)
        t2.fc = nn.Linear(t2.fc.in_features, 7)
        t2.load_state_dict(torch.load(self.config.teacher_models['t2'], 
                                    map_location=DEVICE))
        t2.eval()
        self.teachers['t2'] = t2.to(DEVICE)
        
        # Teacher 3: All classes (ResNet101)
        t3 = models.resnet101(pretrained=True)
        t3.fc = nn.Linear(t3.fc.in_features, NUM_CLASSES)
        t3.load_state_dict(torch.load(self.config.teacher_models['t3'], 
                                    map_location=DEVICE))
        t3.eval()
        self.teachers['t3'] = t3.to(DEVICE)
        
        # Setup student models
        self.students = []
        
        # Student 1: MobileNetV2
        s1 = models.mobilenet_v2(pretrained=False)
        s1.classifier[1] = nn.Linear(s1.classifier[1].in_features, NUM_CLASSES)
        self.students.append(s1.to(DEVICE))
        
        # Student 2: ShuffleNetV2
        s2 = models.shufflenet_v2_x1_0(pretrained=False)
        s2.fc = nn.Linear(s2.fc.in_features, NUM_CLASSES)
        self.students.append(s2.to(DEVICE))
        
        # Student 3: MobileNetV2
        s3 = models.mobilenet_v2(pretrained=False)
        s3.classifier[1] = nn.Linear(s3.classifier[1].in_features, NUM_CLASSES)
        self.students.append(s3.to(DEVICE))
        
        # Student 4: ShuffleNetV2
        s4 = models.shufflenet_v2_x1_0(pretrained=False)
        s4.fc = nn.Linear(s4.fc.in_features, NUM_CLASSES)
        self.students.append(s4.to(DEVICE))
        
        print("Models loaded successfully!")
    
    def setup_training(self):
        """Setup loss functions, optimizers, and schedulers."""
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizers
        self.optimizers = []
        self.schedulers = []
        
        for i, student in enumerate(self.students):
            student_key = f's{i+1}'
            
            optimizer = optim.SGD(
                student.parameters(),
                lr=self.config.learning_rates[student_key],
                weight_decay=self.config.weight_decays[student_key],
                momentum=self.config.momentum
            )
            
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', patience=self.config.patience, verbose=True
            )
            
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
    
    def kld_loss_with_temperature(self, student_logits: torch.Tensor, 
                                teacher_logits: torch.Tensor, 
                                temperature: float) -> torch.Tensor:
        """Compute KL divergence loss with temperature scaling."""
        teacher_logits = teacher_logits / temperature
        kld_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return torch.where(torch.isnan(kld_loss), torch.tensor(0.0), kld_loss)
    
    def compute_distillation_loss(self, outputs: torch.Tensor, 
                                t1_logits: torch.Tensor,
                                t2_logits: torch.Tensor, 
                                t7_logits: torch.Tensor,
                                labels: torch.Tensor) -> torch.Tensor:
        """Compute combined distillation loss from multiple teachers."""
        # Split outputs based on label ranges
        mask1 = labels < 7
        mask2 = (labels >= 7) & (labels < 14)
        
        # Compute specialist teacher losses
        part1 = torch.tensor(0.0)
        part2 = torch.tensor(0.0)
        
        if mask1.any():
            part1 = self.kld_loss_with_temperature(
                outputs[mask1][:, :7], t1_logits, self.config.temperature
            )
        
        if mask2.any():
            part2 = self.kld_loss_with_temperature(
                outputs[mask2][:, 7:14], t2_logits, self.config.temperature
            )
        
        # General teacher loss
        part3 = self.kld_loss_with_temperature(
            outputs, t7_logits, self.config.temperature
        )
        
        # Combine losses
        specialist_loss = part1 + part2
        general_loss = part3
        
        total_loss = (self.config.distillation_weights['specialist'] * specialist_loss + 
                     self.config.distillation_weights['general'] * general_loss)
        
        return torch.where(torch.isnan(total_loss), torch.tensor(0.0), total_loss)
    
    def train_epoch(self) -> Tuple[List[float], List[float]]:
        """Train for one epoch."""
        # Set models to training mode
        for student in self.students:
            student.train()
        
        # Initialize metrics
        train_losses = [0.0] * 4
        correct_counts = [0] * 4
        total_counts = [0] * 4
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass through students
            student_outputs = [student(inputs) for student in self.students]
            
            # Forward pass through teachers
            with torch.no_grad():
                t7_logits = self.teachers['t3'](inputs)
                
                # Get teacher outputs for label subsets
                mask1 = labels < 7
                mask2 = (labels >= 7) & (labels < 14)
                
                t1_logits = self.teachers['t1'](inputs[mask1]) if mask1.any() else None
                t2_logits = self.teachers['t2'](inputs[mask2]) if mask2.any() else None
            
            # Compute losses for each student
            losses = []
            for i, outputs in enumerate(student_outputs):
                # Classification loss
                ce_loss = self.criterion(outputs, labels)
                
                # Distillation loss
                dist_loss = self.compute_distillation_loss(
                    outputs, t1_logits, t2_logits, t7_logits, labels
                )
                
                # Ensemble loss
                ensemble_output = sum(student_outputs) / len(student_outputs)
                ensemble_loss = self.criterion(ensemble_output, labels)
                
                # Combined loss
                student_key = f's{i+1}'
                ensemble_weight = self.config.ensemble_weights[student_key]
                
                total_loss = ce_loss + dist_loss + ensemble_weight * ensemble_loss
                losses.append(total_loss)
                
                train_losses[i] += total_loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total_counts[i] += labels.size(0)
                correct_counts[i] += predicted.eq(labels).sum().item()
            
            # Backward pass and optimization
            for i, (loss, optimizer) in enumerate(zip(losses, self.optimizers)):
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss = sum(train_losses) / len(train_losses) / (batch_idx + 1)
                progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # Calculate average losses and accuracies
        train_losses = [loss / len(self.train_loader) for loss in train_losses]
        train_accuracies = [100 * correct / total for correct, total in 
                           zip(correct_counts, total_counts)]
        
        return train_losses, train_accuracies
    
    def validate_epoch(self) -> Tuple[List[float], List[float], float]:
        """Validate for one epoch."""
        # Set models to evaluation mode
        for student in self.students:
            student.eval()
        
        val_losses = [0.0] * 4
        correct_counts = [0] * 4
        total_counts = [0] * 4
        
        ensemble_correct = 0
        ensemble_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Validation'):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass through students
                student_outputs = [student(inputs) for student in self.students]
                
                # Calculate individual losses and accuracies
                for i, outputs in enumerate(student_outputs):
                    loss = self.criterion(outputs, labels)
                    val_losses[i] += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total_counts[i] += labels.size(0)
                    correct_counts[i] += predicted.eq(labels).sum().item()
                
                # Calculate ensemble accuracy
                ensemble_output = sum(student_outputs) / len(student_outputs)
                _, ensemble_pred = ensemble_output.max(1)
                ensemble_total += labels.size(0)
                ensemble_correct += ensemble_pred.eq(labels).sum().item()
        
        # Calculate average losses and accuracies
        val_losses = [loss / len(self.test_loader) for loss in val_losses]
        val_accuracies = [100 * correct / total for correct, total in 
                         zip(correct_counts, total_counts)]
        ensemble_accuracy = 100 * ensemble_correct / ensemble_total
        
        return val_losses, val_accuracies, ensemble_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_states': [student.state_dict() for student in self.students],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'scheduler_states': [sch.state_dict() for sch in self.schedulers],
            'best_val_losses': self.best_val_losses,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'kl_losses': self.kl_losses,
            'config': self.config.__dict__
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            self.current_epoch = checkpoint['epoch']
            
            # Load student model states
            for i, state_dict in enumerate(checkpoint['student_states']):
                self.students[i].load_state_dict(state_dict)
            
            # Load optimizer states
            for i, state_dict in enumerate(checkpoint['optimizer_states']):
                self.optimizers[i].load_state_dict(state_dict)
            
            # Load scheduler states
            for i, state_dict in enumerate(checkpoint['scheduler_states']):
                self.schedulers[i].load_state_dict(state_dict)
            
            # Load training history
            self.best_val_losses = checkpoint['best_val_losses']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.kl_losses = checkpoint['kl_losses']
            
            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from epoch {self.current_epoch + 1}")
            return True
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def calculate_ensemble_accuracy(self) -> float:
        """Calculate ensemble accuracy on test set."""
        if not all(model is not None for model in self.best_models):
            return 0.0
        
        for model in self.best_models:
            model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = sum(model(inputs) for model in self.best_models) / 4.0
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100 * correct / total
    
    def save_best_models(self):
        """Save individual best models."""
        model_names = ['mnet1.pth', 'snet1.pth', 'mnet2.pth', 'snet2.pth']
        
        for i, (model, name) in enumerate(zip(self.best_models, model_names)):
            if model is not None:
                save_path = os.path.join(self.config.output_dir, name)
                torch.save(model.state_dict(), save_path)
                print(f"Best model {i+1} saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training losses
        for i in range(4):
            axes[0, 0].plot(self.train_losses[i], label=f'Student {i+1}')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot validation losses
        for i in range(4):
            axes[0, 1].plot(self.val_losses[i], label=f'Student {i+1}')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot KL losses (if available)
        if self.kl_losses:
            axes[1, 0].plot(self.kl_losses)
            axes[1, 0].set_title('KL Divergence Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('KL Loss')
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.plots_dir, 'training_curves.jpg')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Main training loop."""
        print(f"Starting training on device: {DEVICE}")
        print(f"Total epochs: {self.config.num_epochs}")
        
        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses, train_accuracies = self.train_epoch()
            
            # Validation phase
            val_losses, val_accuracies, ensemble_accuracy = self.validate_epoch()
            
            # Update learning rate schedulers
            for i, scheduler in enumerate(self.schedulers):
                scheduler.step(val_losses[i])
            
            # Update training history
            for i in range(4):
                self.train_losses[i].append(train_losses[i])
                self.val_losses[i].append(val_losses[i])
            
            # Save best models
            models_updated = []
            for i in range(4):
                if val_losses[i] < self.best_val_losses[i]:
                    self.best_val_losses[i] = val_losses[i]
                    self.best_models[i] = self.students[i]
                    models_updated.append(i+1)
            
            # Calculate ensemble accuracy with best models
            best_ensemble_acc = self.calculate_ensemble_accuracy()
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 70)
            for i in range(4):
                print(f"S{i+1} - Loss: {train_losses[i]:.4f}, "
                     f"Val Loss: {val_losses[i]:.4f}, "
                     f"Val Acc: {val_accuracies[i]:.3f}%")
            
            print(f"Ensemble Acc: {ensemble_accuracy:.3f}%, "
                 f"Best Ensemble: {best_ensemble_acc:.3f}%")
            
            if models_updated:
                print(f"Models updated: {models_updated}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                is_best = len(models_updated) > 0
                self.save_checkpoint(epoch, is_best)
        
        # Final save
        self.save_best_models()
        self.save_checkpoint(self.config.num_epochs - 1, is_best=True)
        self.plot_training_curves()
        
        print("\nTraining completed!")
        print(f"Final ensemble accuracy: {best_ensemble_acc:.3f}%")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Multi-Teacher Knowledge Distillation Training'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint file to resume training from'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file (JSON)'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for training'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Load config from file if specified
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(config)
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()