"""
training and evaluation system for microvision.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

class Trainer:
    """
    Training and evaluation class for microvision models.
    """
    def __init__(self, model: nn.Module, config, device: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = device or config.device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Setup logging
        self.writer = None
        self.setup_logging()
        
    def setup_logging(self):
        """Setup TensorBoard logging."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.config.log_dir)
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing training history
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"New best validation accuracy: {val_acc:.2f}%")
        
        # Final results
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            is_best: Whether this is the best model
        """
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, os.path.join(self.config.ckpt_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.ckpt_dir, 'best_model.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (loss, accuracy, predictions)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                all_preds.extend(pred.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, np.array(all_preds)
