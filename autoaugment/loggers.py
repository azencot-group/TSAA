import numpy as np
import pandas as pd
import torch
import os
import pickle
import time
import random
import itertools
import optuna
import shutil


class CheckpointLogger:
    """
    Handles checkpoint management for model training, including saving and loading
    model states, tracking metrics, and managing early stopping.
    """
    
    def __init__(self, args, last_checkpoint=None, output_dir=None, setting=None):
        """
        Initialize checkpoint logger with configuration options
        
        Args:
            args: Configuration arguments
            last_checkpoint: Path to the last checkpoint (optional)
            output_dir: Directory for saving checkpoints (optional)
            setting: Custom setting identifier (optional)
        """
        self.args = args
        self.output_dir = output_dir

        # Set output directory based on model and task configuration
        if self.output_dir is None:
            if setting is None:
                if self.args.model == 'FEDformer':
                    self.output_dir = (
                        f'./checkpoints_aa/{self.args.features}/{self.args.model}_{self.args.version}/'
                        f'{self.args.task_id}_{self.args.pred_len}_{self.args.random_seed_aa}_{self.args.n_aug_ratio}/'
                    )
                else:
                    self.output_dir = (
                        f'./checkpoints_aa/{self.args.features}/{self.args.model}/'
                        f'{self.args.task_id}_{self.args.pred_len}_{self.args.random_seed_aa}_{self.args.n_aug_ratio}/'
                    )
            else:
                self.output_dir = f'./checkpoints_aa/{setting}/'
        
        self.checkpoint_loc = {}
        self.last_checkpoint = last_checkpoint
        self.n_aug_checkpoint = last_checkpoint

        # Tracking metrics for different phases
        # Epoch-level losses
        self.val_loss = []
        self.test_loss = []
        self.train_loss = []

        # Step-level losses
        self.step_val_loss = []
        self.step_test_loss = []
        self.step_train_loss = []

        # Losses without early stopping (for inference)
        self.val_loss_ex_es = []
        self.test_loss_ex_es = []
        self.train_loss_ex_es = []

        # Model configuration
        self.n_aug_ratio = args.n_aug_ratio
        self.base_checkpoint_path = None
        self.max_epoch = 0
        self.save_flag = True
        self.trial = None  # For Optuna hyperparameter optimization

        # Timing metrics
        self.total_save_time = 0
        self.total_time = 0

    def log_epoch_loss(self, train_loss=None, val_loss=None, test_loss=None):
        """Record losses at the end of each epoch"""
        if val_loss is not None: 
            self.val_loss.append(val_loss)
        if test_loss is not None:
            self.test_loss.append(test_loss)
        if train_loss is not None:
            self.train_loss.append(train_loss)

    def log_step_loss(self, train_loss=None, val_loss=None, test_loss=None):
        """Record losses at each training step"""
        if val_loss is not None: 
            self.step_val_loss.append(val_loss)
        if test_loss is not None:
            self.step_test_loss.append(test_loss)
        if train_loss is not None:
            self.train_loss.append(train_loss)

    def create_checkpoint(self, model, model_optim, epoch,
                         train_loss=None, val_loss=None, test_loss=None,
                         early_stopping=None, scheduler=None, 
                         loader_generator=None, step=False, log_step=False):
        """
        Create a checkpoint with current model state and metrics
        
        Args:
            model: The model to checkpoint
            model_optim: Optimizer state
            epoch: Current epoch number
            train_loss, val_loss, test_loss: Current loss metrics
            early_stopping: Early stopping state
            scheduler: Learning rate scheduler
            loader_generator: Data loader generator
            step: Current step (for step-level checkpointing)
            log_step: Whether to log step-level metrics
        """
        # Record metrics
        self.log_epoch_loss(train_loss, val_loss, test_loss)
        if log_step:
            self.log_step_loss(train_loss, val_loss, test_loss)

        # Handle Optuna hyperparameter optimization if active
        if self.trial is not None:
            n_aug_epoch = int((1 - self.n_aug_ratio) * self.max_epoch)
            self.trial.report(val_loss, epoch - n_aug_epoch)
            self.trial.set_user_attr("val_loss", val_loss)
            
            # Stop if trial should be pruned
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Save checkpoint if enabled
        if self.save_flag:
            self.save_checkpoint(model, model_optim, epoch, 
                               loader_generator, early_stopping, scheduler, step)

    def save_checkpoint(self, model, model_optim, epoch, loader_generator=None,  
                       early_stopping=None, scheduler=None, step=None):
        """
        Save the model checkpoint to disk
        
        Args:
            model: The model to save
            model_optim: Optimizer state
            epoch: Current epoch number
            loader_generator: Data loader generator
            early_stopping: Early stopping state
            scheduler: Learning rate scheduler
            step: Current step (for step-level checkpointing)
        """
        print("Saving checkpoint...")
        start_save_time = time.time()
        
        # Save random states for reproducibility
        random_states = {
            "np": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.random.get_rng_state(),
            "random": random.getstate(),
            "loader_generator_state": loader_generator.get_state() if loader_generator is not None else None
        }
        
        # Create checkpoint dictionary with all state
        checkpoint = {
            "model": model.state_dict(),
            "model_optim": model_optim.state_dict(),
            "epoch": epoch,
            "step": step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "test_loss": self.test_loss,
            "step_train_loss": self.step_train_loss,
            "step_val_loss": self.step_val_loss,
            "step_test_loss": self.step_test_loss,
            "early_stopping": early_stopping,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "random_states": random_states 
        }

        # Create directory if needed and save checkpoint
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.output_dir + f'epoch={epoch}.pth'
        torch.save(checkpoint, path)
        self.checkpoint_loc[epoch] = path 
        
        end_time = time.time()
        self.total_save_time += end_time - start_save_time
        print(f"Checkpoint saved in {end_time - start_save_time:.2f} seconds")

    def load_checkpoint(self, model, model_optim, scheduler=None, loader_generator=None):
        """
        Load a model checkpoint
        
        Args:
            model: Model to load weights into
            model_optim: Optimizer to load state into
            scheduler: Learning rate scheduler
            loader_generator: Data loader generator
            
        Returns:
            tuple: (model, optimizer, epoch, max_epochs, early_stopping, scheduler)
        """
        self.save_flag = False
        print("Loading checkpoint...")
        print(f"n_aug_ratio: {self.n_aug_ratio}")
        
        # Calculate epoch based on augmentation ratio
        n_aug_epoch = int((1 - self.n_aug_ratio) * self.max_epoch)
        path = self.checkpoint_loc[n_aug_epoch]

        # Load checkpoint data
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        model_optim.load_state_dict(checkpoint['model_optim'])
        base_epoch = checkpoint['epoch']

        # Restore random states for reproducibility
        if loader_generator is not None:
            loader_generator.set_state(checkpoint['random_states']['loader_generator_state'])
            
        torch.manual_seed(0).set_state(checkpoint['random_states']['torch'])
        # Note: this must come after setting torch random state
        torch.cuda.set_rng_state(checkpoint['random_states']['torch_cuda'])
        np.random.set_state(checkpoint['random_states']['np'])
        random.setstate(checkpoint['random_states']['random'])
        
        max_epochs = self.max_epoch

        # Restore tracked metrics
        print(f"Max epoch in load_checkpoint: {self.max_epoch}")
        self.val_loss = checkpoint['val_loss']
        self.test_loss = checkpoint['test_loss']
        self.train_loss = checkpoint['train_loss']
                
        self.step_val_loss = checkpoint['step_val_loss']
        self.step_test_loss = checkpoint['step_test_loss']
        self.step_train_loss = checkpoint['step_train_loss']

        # Restore other states
        es = checkpoint['early_stopping']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            scheduler = None
    
        print("Checkpoint loaded")
        print(f"Base epoch: {base_epoch}")
        print(f"Train loss: {self.train_loss}")
        print(f"Test loss: {self.test_loss}")
        print(f"Val loss: {self.val_loss}")
        
        return model, model_optim, base_epoch, max_epochs, es, scheduler

    def finalize_training(self, max_epoch, start_time):
        """
        Finalize training metrics and timing information
        
        Args:
            max_epoch: Maximum epoch reached
            start_time: Training start time
        """
        self.total_time = time.time() - start_time - self.total_save_time
        self.total_save_time = 0
        self.max_epoch = max_epoch
        
        # Save full history before truncating
        self.val_loss_ex_es = self.val_loss
        self.test_loss_ex_es = self.test_loss
        self.train_loss_ex_es = self.train_loss

        # Truncate history to max epoch (handles early stopping)
        self.val_loss = self.val_loss[:max_epoch]
        self.test_loss = self.test_loss[:max_epoch]
        self.train_loss = self.train_loss[:max_epoch]

    def remove_checkpoint_dir(self, path=None):
        """
        Remove checkpoint directory and all contained files
        
        Args:
            path: Path to directory (uses default output_dir if None)
        """
        if path is None:
            path = self.output_dir
            
        if os.path.exists(path):
            shutil.rmtree(path)