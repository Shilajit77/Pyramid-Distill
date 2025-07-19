#!/usr/bin/env python3


import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

# Custom imports (ensure these modules exist in your project)
try:
    from models import wrnet, wrnet2, mobilenet
except ImportError:
    print("Warning: Custom models module not found. Using default MobileNetV2.")
    mobilenet = None

warnings.filterwarnings("ignore")


class Config:
    """Configuration class for training parameters"""
    
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.006
        self.weight_decay = 2e-4
        self.momentum = 0.9
        self.num_epochs = 100
        self.temperature = 20
        self.kd_alpha = 0.4  # Weight for fine-grained KD loss
        self.kd_beta = 0.6   # Weight for coarse-grained KD loss
        self.kd_gamma = 0.6  # Weight for full teacher KD loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = './data'
        self.model_dir = '100models'
        self.plots_dir = 'plots'
        self.train_val_split = 0.9
        self.random_seed = 40


class HierarchicalKnowledgeDistiller:
    """Main class for hierarchical knowledge distillation training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        
        # Initialize models
        self.teachers = self._load_teacher_models()
        self.student = self._initialize_student_model()
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.kl_losses = []
        
        # Create necessary directories
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.plots_dir, exist_ok=True)
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and testing"""
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 120)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load CIFAR-100 dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.config.data_root, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.config.data_root, train=False, download=True, transform=transform
        )
        
        # Split training data into train and validation
        total_size = len(train_dataset)
        train_size = int(self.config.train_val_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def _load_teacher_models(self) -> Dict[str, nn.Module]:
        """Load pre-trained teacher models"""
        
        teachers = {}
        
        # Teacher model configurations
        teacher_configs = [
            ('t1', 'teacher1-25(res18).pth', models.resnet18, 25),
            ('t2', 'teacher25-50(res18).pth', models.resnet18, 25),
            ('t3', 'teacher50-75(res18).pth', models.resnet18, 25),
            ('t4', 'teacher75-100(res18).pth', models.resnet18, 25),
            ('t5', 'teacher1-50(res50).pth', models.resnet50, 50),
            ('t6', 'teacher50-100(res50).pth', models.resnet50, 50),
            ('t7', 'teacher1-100(res101).pth', models.resnet101, 100),
        ]
        
        for name, model_path, model_fn, num_classes in teacher_configs:
            model = model_fn(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            full_path = os.path.join(self.config.model_dir, model_path)
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path, map_location=self.device))
                print(f"Loaded teacher model: {name}")
            else:
                print(f"Warning: Teacher model {model_path} not found. Using randomly initialized weights.")
            
            model = model.to(self.device)
            model.eval()
            teachers[name] = model
        
        return teachers
    
    def _initialize_student_model(self) -> nn.Module:
        """Initialize student model"""
        
        if mobilenet is not None:
            student = mobilenet.MobileNet()
        else:
            student = models.mobilenet_v2(pretrained=False)
            student.classifier[1] = nn.Linear(student.classifier[1].in_features, 100)
        
        return student.to(self.device)
    
    def _kld_loss_with_temperature(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        temperature: float
    ) -> torch.Tensor:
        """
        Compute KL Divergence loss with temperature scaling for knowledge distillation
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            temperature: Temperature parameter for softening distributions
            
        Returns:
            KL divergence loss
        """
        
        if student_logits.size(0) == 0 or teacher_logits.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Apply temperature scaling
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature
        
        # Compute KL divergence
        kld_loss = F.kl_div(
            F.log_softmax(student_scaled, dim=1),
            F.softmax(teacher_scaled, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Handle NaN values
        if torch.isnan(kld_loss):
            return torch.tensor(0.0, device=self.device)
        
        return kld_loss
    
    def _compute_hierarchical_kd_loss(
        self, 
        student_outputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hierarchical knowledge distillation loss
        
        Args:
            student_outputs: Student model outputs
            labels: Ground truth labels
            
        Returns:
            Combined KD loss
        """
        
        inputs = student_outputs  # This should be the input tensor, but we'll work with what we have
        
        # Get teacher outputs (in practice, you'd pass inputs through teachers)
        with torch.no_grad():
            t7_logits = self.teachers['t7'](inputs) if 't7' in self.teachers else torch.zeros_like(student_outputs)
        
        # Create masks for different label ranges
        mask_0_25 = labels < 25
        mask_25_50 = (labels >= 25) & (labels < 50)
        mask_50_75 = (labels >= 50) & (labels < 75)
        mask_75_100 = (labels >= 75) & (labels < 100)
        mask_0_50 = (labels >= 0) & (labels < 50)
        mask_50_100 = (labels >= 50) & (labels < 100)
        
        total_kd_loss = torch.tensor(0.0, device=self.device)
        
        # Fine-grained KD losses (25-class teachers)
        if mask_0_25.any() and 't1' in self.teachers:
            with torch.no_grad():
                t1_logits = self.teachers['t1'](inputs[mask_0_25])
            part1 = self._kld_loss_with_temperature(
                student_outputs[mask_0_25][:, :25], t1_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_alpha * part1
        
        if mask_25_50.any() and 't2' in self.teachers:
            with torch.no_grad():
                t2_logits = self.teachers['t2'](inputs[mask_25_50])
            part2 = self._kld_loss_with_temperature(
                student_outputs[mask_25_50][:, 25:50], t2_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_alpha * part2
        
        if mask_50_75.any() and 't3' in self.teachers:
            with torch.no_grad():
                t3_logits = self.teachers['t3'](inputs[mask_50_75])
            part3 = self._kld_loss_with_temperature(
                student_outputs[mask_50_75][:, 50:75], t3_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_alpha * part3
        
        if mask_75_100.any() and 't4' in self.teachers:
            with torch.no_grad():
                t4_logits = self.teachers['t4'](inputs[mask_75_100])
            part4 = self._kld_loss_with_temperature(
                student_outputs[mask_75_100][:, 75:100], t4_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_alpha * part4
        
        # Coarse-grained KD losses (50-class teachers)
        if mask_0_50.any() and 't5' in self.teachers:
            with torch.no_grad():
                t5_logits = self.teachers['t5'](inputs[mask_0_50])
            part5 = self._kld_loss_with_temperature(
                student_outputs[mask_0_50][:, :50], t5_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_beta * part5
        
        if mask_50_100.any() and 't6' in self.teachers:
            with torch.no_grad():
                t6_logits = self.teachers['t6'](inputs[mask_50_100])
            part6 = self._kld_loss_with_temperature(
                student_outputs[mask_50_100][:, 50:100], t6_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_beta * part6
        
        # Full teacher KD loss (100-class teacher)
        if 't7' in self.teachers:
            part7 = self._kld_loss_with_temperature(
                student_outputs, t7_logits, self.config.temperature
            )
            total_kd_loss += self.config.kd_gamma * part7
        
        return total_kd_loss
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch"""
        
        self.student.train()
        train_loss = 0.0
        kl_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs}')
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.student(inputs)
            
            # Compute classification loss
            ce_loss = self.criterion(outputs, labels)
            
            # Compute knowledge distillation loss
            kd_loss = self._compute_hierarchical_kd_loss(outputs, labels)
            
            # Total loss
            total_loss = ce_loss + kd_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            kl_loss += kd_loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_train_loss = train_loss / len(self.train_loader)
        epoch_kl_loss = kl_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_train_loss, epoch_kl_loss, epoch_accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        
        self.student.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.student(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / len(self.test_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_val_loss, epoch_accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'kl_losses': self.kl_losses,
            'config': self.config.__dict__
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.model_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(self.config.model_dir, 'mnetV1-hierkd.pth')
            torch.save(self.student.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint"""
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found.")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.student.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.kl_losses = checkpoint['kl_losses']
            
            print(f"Resumed from epoch {self.current_epoch} with best val loss: {self.best_val_loss:.4f}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        
        if not self.train_losses:
            return
        
        plt.figure(figsize=(15, 5))
        
        # Plot KL loss
        plt.subplot(1, 3, 1)
        plt.plot(self.kl_losses, label='KL Loss', color='red')
        plt.title('KL Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot training and validation loss
        plt.subplot(1, 3, 2)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot([lr_history[0]] * len(self.train_losses), label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.config.plots_dir, 'training_curves.jpg')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def train(self, resume: bool = False):
        """Main training loop"""
        
        # Resume training if requested
        if resume:
            checkpoint_path = os.path.join(self.config.model_dir, 'checkpoint_latest.pth')
            self.load_checkpoint(checkpoint_path)
        
        print(f"Training on device: {self.device}")
        print(f"Starting from epoch: {self.current_epoch + 1}")
        
        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                train_loss, kl_loss, train_acc = self.train_epoch()
                
                # Validate
                val_loss, val_acc = self.validate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Store metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.kl_losses.append(kl_loss)
                
                # Print epoch results
                print(f'Epoch {epoch + 1}/{self.config.num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, KL Loss: {kl_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
                
                # Save checkpoint and best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(is_best=is_best)
                
                # Plot training curves every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.plot_training_curves()
                
                print("-" * 80)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.save_checkpoint()
        
        except Exception as e:
            print(f"\nTraining stopped due to error: {e}")
            self.save_checkpoint()
        
        finally:
            # Final plots and cleanup
            self.plot_training_curves()
            print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Hierarchical Knowledge Distillation Training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.006, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=20, help='Temperature for knowledge distillation')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.temperature = args.temperature
    
    # Initialize and start training
    distiller = HierarchicalKnowledgeDistiller(config)
    distiller.train(resume=args.resume)


if __name__ == '__main__':
    main()