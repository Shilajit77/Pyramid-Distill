

import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score

from models import wrnet, wrnet2, mobilenet

warnings.filterwarnings("ignore")

class Config:
    """Configuration class for training parameters"""
    
    # Data parameters
    BATCH_SIZE = 32
    NUM_CLASSES = 100
    TRAIN_SPLIT = 0.9
    RANDOM_SEED = 40
    
    # Training parameters
    NUM_EPOCHS = 500
    TEMPERATURE = 20
    ENSEMBLE_ALPHA = 0.2  # Weight for ensemble loss in student 1 and 3
    ENSEMBLE_BETA = 0.8   # Weight for ensemble loss in student 2 and 4
    
    # Optimizer parameters
    LR_STUDENT_1 = 0.06
    LR_STUDENT_2 = 0.08
    LR_STUDENT_3 = 0.06
    LR_STUDENT_4 = 0.08
    WEIGHT_DECAY = 2e-4
    MOMENTUM = 0.9
    SCHEDULER_PATIENCE = 3
    
    # Paths
    DATA_ROOT = './data'
    TEACHER_MODELS_PATH = '/csehome/m22cs062/cifar100/100models'
    ENSEMBLE_MODELS_PATH = 'ensemble_models'
    PLOTS_PATH = 'plots'
    CHECKPOINT_PATH = 'checkpoints'


class KnowledgeDistillationTrainer:
    """Main trainer class for knowledge distillation"""
    
    def __init__(self, config: Config, device: torch.device, resume_from: Optional[str] = None):
        self.config = config
        self.device = device
        self.resume_from = resume_from
        
        # Create directories
        os.makedirs(config.ENSEMBLE_MODELS_PATH, exist_ok=True)
        os.makedirs(config.PLOTS_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_data()
        
        # Initialize models
        self.teacher_models = self._load_teacher_models()
        self.student_models, self.optimizers, self.schedulers = self._setup_student_models()
        
        # Training state
        self.start_epoch = 0
        self.best_val_losses = [float('inf')] * 4
        self.best_models = [None] * 4
        self.train_losses_history = [[] for _ in range(4)]
        self.val_losses_history = [[] for _ in range(4)]
        self.kl_losses_history = []
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Resume training if checkpoint provided
        if self.resume_from:
            self._load_checkpoint(self.resume_from)
    
    def _setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and testing"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 120)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.config.DATA_ROOT, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.config.DATA_ROOT, train=False, download=True, transform=transform
        )
        
        # Split training data into train/validation
        total_size = len(train_dataset)
        train_size = int(self.config.TRAIN_SPLIT * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.RANDOM_SEED)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False
        )
        
        print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def _load_teacher_models(self) -> Dict[str, nn.Module]:
        """Load pre-trained teacher models"""
        teacher_configs = [
            ('t1', 'resnet18', 25, 'teacher1-25(res18).pth'),
            ('t2', 'resnet18', 25, 'teacher25-50(res18).pth'),
            ('t3', 'resnet18', 25, 'teacher50-75(res18).pth'),
            ('t4', 'resnet18', 25, 'teacher75-100(res18).pth'),
            ('t5', 'resnet50', 50, 'teacher1-50(res50).pth'),
            ('t6', 'resnet50', 50, 'teacher50-100(res50).pth'),
            ('t7', 'resnet101', 100, 'teacher1-100(res101).pth'),
        ]
        
        teachers = {}
        for name, arch, num_classes, filename in teacher_configs:
            if arch == 'resnet18':
                model = models.resnet18(pretrained=True)
            elif arch == 'resnet50':
                model = models.resnet50(pretrained=True)
            elif arch == 'resnet101':
                model = models.resnet101(pretrained=True)
            
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(
                os.path.join(self.config.TEACHER_MODELS_PATH, filename)
            ))
            model.to(self.device)
            model.eval()
            teachers[name] = model
            print(f"Loaded teacher model: {name}")
        
        return teachers
    
    def _setup_student_models(self) -> Tuple[List[nn.Module], List[optim.Optimizer], List[ReduceLROnPlateau]]:
        """Setup student models, optimizers, and schedulers"""
        # Initialize student models
        students = [
            mobilenet.MobileNet(),
            models.shufflenet_v2_x1_0(pretrained=False),
            mobilenet.MobileNet(),
            models.shufflenet_v2_x1_0(pretrained=False)
        ]
        
        # Modify output layers for ShuffleNet models
        students[1].fc = nn.Linear(students[1].fc.in_features, self.config.NUM_CLASSES)
        students[3].fc = nn.Linear(students[3].fc.in_features, self.config.NUM_CLASSES)
        
        # Move to device
        for student in students:
            student.to(self.device)
        
        # Setup optimizers
        learning_rates = [
            self.config.LR_STUDENT_1, self.config.LR_STUDENT_2,
            self.config.LR_STUDENT_3, self.config.LR_STUDENT_4
        ]
        
        optimizers = []
        schedulers = []
        
        for i, (student, lr) in enumerate(zip(students, learning_rates)):
            optimizer = optim.SGD(
                student.parameters(), lr=lr,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=self.config.MOMENTUM
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min',
                patience=self.config.SCHEDULER_PATIENCE,
                verbose=True
            )
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            print(f"Setup student model {i+1} with LR: {lr}")
        
        return students, optimizers, schedulers
    
    def kld_loss_with_temperature(self, student_logits: torch.Tensor,
                                teacher_logits: torch.Tensor,
                                temperature: float) -> torch.Tensor:
        """Calculate KL divergence loss with temperature scaling"""
        teacher_logits = teacher_logits / temperature
        
        kld_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return torch.where(torch.isnan(kld_loss), torch.tensor(0.0), kld_loss)
    
    def compute_distillation_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-teacher distillation loss"""
        # Filter inputs and get teacher outputs based on label ranges
        mask_0_25 = labels < 25
        mask_25_50 = (labels >= 25) & (labels < 50)
        mask_50_75 = (labels >= 50) & (labels < 75)
        mask_75_100 = (labels >= 75) & (labels < 100)
        mask_0_50 = (labels >= 0) & (labels < 50)
        mask_50_100 = (labels >= 50) & (labels < 100)
        
        distill_loss = 0.0
        
        # Individual teacher losses (25 classes each)
        if mask_0_25.any():
            t1_logits = self.teacher_models['t1'](torch.masked_select(
                inputs, mask_0_25.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part1 = self.kld_loss_with_temperature(
                outputs[mask_0_25][:, :25], t1_logits, self.config.TEMPERATURE
            )
            distill_loss += part1
        
        if mask_25_50.any():
            t2_logits = self.teacher_models['t2'](torch.masked_select(
                inputs, mask_25_50.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part2 = self.kld_loss_with_temperature(
                outputs[mask_25_50][:, 25:50], t2_logits, self.config.TEMPERATURE
            )
            distill_loss += part2
        
        if mask_50_75.any():
            t3_logits = self.teacher_models['t3'](torch.masked_select(
                inputs, mask_50_75.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part3 = self.kld_loss_with_temperature(
                outputs[mask_50_75][:, 50:75], t3_logits, self.config.TEMPERATURE
            )
            distill_loss += part3
        
        if mask_75_100.any():
            t4_logits = self.teacher_models['t4'](torch.masked_select(
                inputs, mask_75_100.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part4 = self.kld_loss_with_temperature(
                outputs[mask_75_100][:, 75:100], t4_logits, self.config.TEMPERATURE
            )
            distill_loss += part4
        
        # Broader teacher losses (50 classes each)
        if mask_0_50.any():
            t5_logits = self.teacher_models['t5'](torch.masked_select(
                inputs, mask_0_50.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part5 = self.kld_loss_with_temperature(
                outputs[mask_0_50][:, :50], t5_logits, self.config.TEMPERATURE
            )
            distill_loss += 0.6 * part5
        
        if mask_50_100.any():
            t6_logits = self.teacher_models['t6'](torch.masked_select(
                inputs, mask_50_100.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ).view(-1, *inputs.shape[1:]))
            part6 = self.kld_loss_with_temperature(
                outputs[mask_50_100][:, 50:100], t6_logits, self.config.TEMPERATURE
            )
            distill_loss += 0.6 * part6
        
        # Full teacher loss (100 classes)
        t7_logits = self.teacher_models['t7'](inputs)
        part7 = self.kld_loss_with_temperature(
            outputs, t7_logits, self.config.TEMPERATURE
        )
        distill_loss += 0.6 * part7
        
        return torch.where(torch.isnan(distill_loss), torch.tensor(0.0), distill_loss)
    
    def evaluate_ensemble(self, models: List[nn.Module]) -> float:
        """Evaluate ensemble accuracy on test set"""
        all_predictions = []
        all_labels = []
        
        for model in models:
            if model is not None:
                model.eval()
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                ensemble_output = torch.zeros_like(self.student_models[0](inputs))
                valid_models = 0
                
                for model in models:
                    if model is not None:
                        outputs = model(inputs)
                        ensemble_output += outputs
                        valid_models += 1
                
                if valid_models > 0:
                    ensemble_output /= valid_models
                
                _, predicted = torch.max(ensemble_output, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        return accuracy
    
    def train_epoch(self, epoch: int) -> Tuple[List[float], List[float], float]:
        """Train for one epoch"""
        # Set models to training mode
        for model in self.student_models:
            model.train()
        
        train_losses = [0.0] * 4
        correct_counts = [0] * 4
        total_counts = [0] * 4
        kl_loss_total = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass through student models
            student_outputs = []
            for model in self.student_models:
                outputs = model(inputs)
                student_outputs.append(outputs)
            
            # Compute losses for each student
            losses = []
            ensemble_output = sum(student_outputs) / len(student_outputs)
            ensemble_loss = self.criterion(ensemble_output, labels)
            
            for i, outputs in enumerate(student_outputs):
                # Compute distillation loss (placeholder - needs actual implementation)
                distill_loss = self.compute_distillation_loss(outputs, labels)
                
                # Compute total loss
                ce_loss = self.criterion(outputs, labels)
                
                if i in [0, 2]:  # Students 1 and 3
                    total_loss = ce_loss + distill_loss + self.config.ENSEMBLE_ALPHA * ensemble_loss
                else:  # Students 2 and 4
                    total_loss = ce_loss + distill_loss + self.config.ENSEMBLE_BETA * ensemble_loss
                
                losses.append(total_loss)
            
            # Backward pass and optimization
            for i, loss in enumerate(losses):
                self.optimizers[i].zero_grad()
                loss.backward(retain_graph=True)
                self.optimizers[i].step()
                
                train_losses[i] += loss.item()
                
                # Calculate accuracy
                _, predicted = student_outputs[i].max(1)
                total_counts[i] += labels.size(0)
                correct_counts[i] += predicted.eq(labels).sum().item()
        
        # Calculate average losses and accuracies
        avg_losses = [loss / len(self.train_loader) for loss in train_losses]
        accuracies = [100 * correct / total for correct, total in zip(correct_counts, total_counts)]
        
        return avg_losses, accuracies, kl_loss_total / len(self.train_loader)
    
    def validate_epoch(self) -> Tuple[List[float], List[float]]:
        """Validate for one epoch"""
        # Set models to evaluation mode
        for model in self.student_models:
            model.eval()
        
        val_losses = [0.0] * 4
        correct_counts = [0] * 4
        total_counts = [0] * 4
        ensemble_correct = 0
        ensemble_total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                student_outputs = []
                for i, model in enumerate(self.student_models):
                    outputs = model(inputs)
                    student_outputs.append(outputs)
                    
                    loss = self.criterion(outputs, labels)
                    val_losses[i] += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total_counts[i] += labels.size(0)
                    correct_counts[i] += predicted.eq(labels).sum().item()
                
                # Ensemble evaluation
                ensemble_output = sum(student_outputs) / len(student_outputs)
                _, ensemble_pred = ensemble_output.max(1)
                ensemble_total += labels.size(0)
                ensemble_correct += ensemble_pred.eq(labels).sum().item()
        
        avg_losses = [loss / len(self.val_loader) for loss in val_losses]
        accuracies = [100 * correct / total for correct, total in zip(correct_counts, total_counts)]
        ensemble_accuracy = 100 * ensemble_correct / ensemble_total
        
        return avg_losses, accuracies, ensemble_accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'student_models': [model.state_dict() for model in self.student_models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sched.state_dict() for sched in self.schedulers],
            'best_val_losses': self.best_val_losses,
            'train_losses_history': self.train_losses_history,
            'val_losses_history': self.val_losses_history,
            'kl_losses_history': self.kl_losses_history,
            'config': self.config.__dict__
        }
        
        checkpoint_file = os.path.join(self.config.CHECKPOINT_PATH, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_file)
        
        if is_best:
            best_file = os.path.join(self.config.CHECKPOINT_PATH, 'best_checkpoint.pth')
            torch.save(checkpoint, best_file)
            print(f"Saved best checkpoint at epoch {epoch + 1}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch']
        
        # Load model states
        for i, state_dict in enumerate(checkpoint['student_models']):
            self.student_models[i].load_state_dict(state_dict)
        
        # Load optimizer states
        for i, state_dict in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(state_dict)
        
        # Load scheduler states
        for i, state_dict in enumerate(checkpoint['schedulers']):
            self.schedulers[i].load_state_dict(state_dict)
        
        # Load training history
        self.best_val_losses = checkpoint['best_val_losses']
        self.train_losses_history = checkpoint['train_losses_history']
        self.val_losses_history = checkpoint['val_losses_history']
        self.kl_losses_history = checkpoint['kl_losses_history']
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        # Plot losses for each student
        for i in range(4):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses_history[i], label=f'Student {i+1} Train Loss')
            plt.plot(self.val_losses_history[i], label=f'Student {i+1} Val Loss')
            plt.title(f'Student {i+1} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.PLOTS_PATH, f'student_{i+1}_curves.png'))
            plt.close()
        
        # Plot KL divergence loss
        if self.kl_losses_history:
            plt.figure(figsize=(8, 6))
            plt.plot(self.kl_losses_history)
            plt.title('KL Divergence Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('KL Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.config.PLOTS_PATH, 'kl_loss_curves.png'))
            plt.close()
    
    def train(self):
        """Main training loop"""
        print("Starting knowledge distillation training...")
        print(f"Training from epoch {self.start_epoch + 1} to {self.config.NUM_EPOCHS}")
        
        is_best = False
        
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            # Training
            train_losses, train_accs, kl_loss = self.train_epoch(epoch)
            
            # Validation
            val_losses, val_accs, ensemble_acc = self.validate_epoch()
            
            # Update learning rate schedulers
            for i, scheduler in enumerate(self.schedulers):
                scheduler.step(val_losses[i])
            
            # Save best models
            for i in range(4):
                if val_losses[i] < self.best_val_losses[i]:
                    self.best_val_losses[i] = val_losses[i]
                    model_path = os.path.join(
                        self.config.ENSEMBLE_MODELS_PATH,
                        f'student_{i+1}_best.pth'
                    )
                    torch.save(self.student_models[i].state_dict(), model_path)
                    self.best_models[i] = self.student_models[i]
                    print(f"Saved best model for student {i+1}")
                    if i == 0:  # Consider it best overall if first student improves
                        is_best = True
            
            # Update history
            for i in range(4):
                self.train_losses_history[i].append(train_losses[i])
                self.val_losses_history[i].append(val_losses[i])
            self.kl_losses_history.append(kl_loss)
            
            # Evaluate ensemble
            if all(model is not None for model in self.best_models):
                best_ensemble_acc = self.evaluate_ensemble(self.best_models)
            else:
                best_ensemble_acc = 0.0
            
            # Print progress
            print(f'\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}:')
            for i in range(4):
                print(f'Student {i+1} - Train Loss: {train_losses[i]:.4f}, '
                      f'Val Loss: {val_losses[i]:.4f}, Val Acc: {val_accs[i]:.3f}%')
            print(f'Current Ensemble Acc: {ensemble_acc:.3f}%, Best Ensemble Acc: {best_ensemble_acc:.3f}%')
            
            # Save checkpoint every 10 epochs or if best
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
                is_best = False
            
            # Plot training curves every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.plot_training_curves()
        
        print("Training completed!")
        self.plot_training_curves()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU ID to use (default: auto-select)')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(args.gpu)
        else:
            device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    # Initialize configuration
    config = Config()
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        config=config,
        device=device,
        resume_from=args.resume
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
