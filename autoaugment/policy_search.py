import numpy as np
import pandas as pd 
import copy
import sys
from exp.exp_main import Exp_Main
from autoaugment.loggers import CheckpointLogger
from autoaugment.augment import AugmentTS
import optuna
import gc
import torch
import os
import random
import logging
import time


class BasePolicy:
    """
    Base class for time series augmentation policy search and evaluation.
    Handles training, testing, and logging of augmentation experiments.
    """
    def __init__(self, args,
                policy=None,  # None for no policy or non-None for a deterministic policy
                method='none',
                pre_stl=True,
                mag_bins=-1,  # -1 for continuous with range [0,1], else mag bins
                aug_params=None,
                baseline_exp=None,
                baseline_logger=None):
        """
        Initialize a base policy object for time series augmentation
        
        Args:
            args: Configuration arguments
            policy: Augmentation policy (None for no policy)
            method: Method name
            pre_stl: Whether to precompute STL decomposition
            mag_bins: Number of magnitude bins (-1 for continuous)
            aug_params: Augmentation parameters
            baseline_exp: Baseline experiment object
            baseline_logger: Baseline logger object
        """
        self.save_baseline = args.save_baseline
        self.args = args
        
        if aug_params is None:
            self.aug_params = {
                'method': method,
                'pre_stl': pre_stl,
                'mag_bins': mag_bins,
                'policy': policy,
            }
        else:
            self.aug_params = aug_params

        self.baseline_exp = baseline_exp
        self.baseline_logger = baseline_logger
        self.final_exp = None
        self.final_policy = None
        self.log_info = {}
        self.setting = None

    def save_logs(self, setting): 
        """
        Save experiment logs and results to disk
        
        Args:
            setting: Experiment configuration identifier
        """
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        run = {}
        run['model_params'] = vars(self.args)
        run['aug_params'] = self.aug_params
        run['final_mse'] = self.log_info['final_mse']
        run['baseline_mse'] = self.log_info['baseline_mse']
        run['final_mae'] = self.log_info['final_mae']
        run['baseline_mae'] = self.log_info['baseline_mae']
        run['final_policy'] = self.final_policy

        run['baseline_train_loss'] = self.log_info['baseline_logger'].train_loss
        run['baseline_val_loss'] = self.log_info['baseline_logger'].val_loss
        run['baseline_test_loss'] = self.log_info['baseline_logger'].test_loss

        run['final_train_loss'] = self.log_info['final_logger'].train_loss
        run['final_val_loss'] = self.log_info['final_logger'].val_loss
        run['final_test_loss'] = self.log_info['final_logger'].test_loss
        
        if self.aug_params['method'] == 'TSAA':
            run['study'] = self.study.trials_dataframe()
            run['trial_info'] = self.study.get_trials()

        torch.save(run, folder_path + f'run_logs_{self.aug_params["method"]}.pt')

    def reset_seed(self, seed):
        """Set all random seeds to ensure reproducibility"""
        fix_seed = seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    def train(self, setting, remove_checkpoints=True):
        """
        Train models with and without augmentation
        
        Args:
            setting: Experiment configuration identifier
            remove_checkpoints: Whether to remove checkpoints after training
        """
        self.setting = setting

        # Train baseline model if not already trained
        if self.baseline_logger is None or self.baseline_exp is None:
            self.baseline_logger = CheckpointLogger(self.args, setting=setting)
            self.baseline_exp = Exp_Main(self.args, self.baseline_logger)
            self.baseline_exp.train(setting, stage2=False, save=self.save_baseline)

        self.log_info['baseline_logger'] = self.baseline_logger

        # Train with augmentation or use baseline based on method
        if self.aug_params['method'] == 'none':
            self.aug_params['policy'] = None
            self.final_exp, final_logger = self.baseline_exp, self.baseline_logger
        else:
            use_early_stopping = False
            if self.aug_params['method'] in ['direct', 'FastAA']:
                use_early_stopping = True
                
            self.final_exp, final_logger = self.fit_method(
                return_exp=True,
                policy=self.aug_params['policy'],
                save=self.args.save_aug,
                use_early_stopping=use_early_stopping
            )
        
        self.log_info['final_logger'] = final_logger

        # Save final policy if different from baseline
        if self.final_exp != self.baseline_exp:
            self.final_policy = self.final_exp.augment_ts.policy

        # Test models and record metrics
        metric_dict_final, metric_dict_baseline = self.test(setting, self.args.save_predict)
        self.log_info['final_mse'] = metric_dict_final['mse']
        self.log_info['baseline_mse'] = metric_dict_baseline['mse']
        self.log_info['final_mae'] = metric_dict_final['mae']
        self.log_info['baseline_mae'] = metric_dict_baseline['mae']

        # Save logs
        self.save_logs(setting)

        # Clean up checkpoints if requested
        if remove_checkpoints:
            self.baseline_logger.remove_checkpoint_dir()
        
    def test(self, setting, save=False):
        """
        Test baseline and final models
        
        Args:
            setting: Experiment configuration identifier
            save: Whether to save predictions
            
        Returns:
            tuple: Metric dictionaries for final and baseline models
        """
        print("Baseline Model")
        metric_dict_baseline = self.baseline_exp.test(setting, save=save)
        print("Final Policy: ", self.final_policy)
        metric_dict_final = self.final_exp.test(setting, autoaugment=True, save=save)
        return metric_dict_final, metric_dict_baseline

    def fit_method(self, policy=None, return_exp=True, trial=None, save=False, use_early_stopping=False):
        """
        Train a model with specified augmentation policy
        
        Args:
            policy: Augmentation policy
            return_exp: Whether to return experiment object
            trial: Optuna trial object (for hyperparameter optimization)
            save: Whether to save model
            use_early_stopping: Whether to use early stopping
            
        Returns:
            Model object or logger object
        """
        print("Fit method, policy:", policy)
        start_time = time.time()
        
        # Initialize augmentation with policy
        augment_ts = AugmentTS(seasonality=self.args.seasonality, aug_params=self.aug_params)
        augment_ts.assign_policy(policy)

        # Setup logger and experiment
        exp_logger = copy.deepcopy(self.baseline_logger)
        if trial is not None:
            exp_logger.trial = trial

        exp = Exp_Main(self.args, exp_logger, augment_ts, use_early_stopping)
        exp.train(self.setting, stage2=True, save=save)
        
        elapsed = time.time() - start_time
        print(f"Policy: {policy}, Time: {elapsed:.2f}s")
        
        if return_exp:
            return exp, exp_logger
        else:
            return exp_logger


class PolicySearch(BasePolicy):
    """Base class for augmentation policy search methods"""
    
    def __init__(self,
                method,
                n_ops=2,
                max_top_k=2,
                aug_params=None,
                *args,
                **kwargs):
        """
        Initialize a policy search object
        
        Args:
            method: Search method name
            n_ops: Number of operations in each sub-policy
            max_top_k: Maximum number of top policies to select
            aug_params: Augmentation parameters
        """
        kwargs['policy'] = None
        kwargs['method'] = method
        kwargs['aug_params'] = aug_params
        super().__init__(*args, **kwargs)
        
        if aug_params is None:
            self.aug_params['n_ops'] = n_ops
            self.aug_params['max_top_k'] = max_top_k
        else:
            if not all(i in aug_params for i in ['n_ops', 'max_top_k']):
                raise Exception('max_top_k or n_ops not defined')
                
        self.aug_params['search_time'] = 0
        self.aug_params['baseline_time'] = 0
        self.aug_params['final_time'] = 0


class RandAugment(PolicySearch):
    """
    RandAugment implementation for time series
    Tries different numbers of operations and magnitudes randomly
    """
    
    def __init__(self,
                mag_bins=8,
                aug_params=None,
                *args,
                **kwargs):
        """
        Initialize RandAugment
        
        Args:
            mag_bins: Number of magnitude bins
            aug_params: Augmentation parameters
        """
        kwargs['method'] = 'RandAugment'
        kwargs['mag_bins'] = mag_bins
        kwargs['aug_params'] = aug_params
        super().__init__(*args, **kwargs)

    def select_best_policy(self, trial_results):
        """
        Select best policies based on validation loss
        
        Args:
            trial_results: List of trial result dictionaries
            
        Returns:
            list: Best policies
        """
        policy_df = pd.DataFrame(trial_results).sort_values(by=['val_loss'])
        top_policies = policy_df.iloc[:self.aug_params['max_top_k']]['policy'].values
        print("top_policies", top_policies)
        top_sub_policies = [p[0] for p in top_policies]
        best_policy = top_sub_policies
        return best_policy

    def fit_method(self, policy=None, return_exp=True, trial=None, save=False, use_early_stopping=False):
        """
        Run random search to find best policies
        
        Args:
            policy: Initial policy (not used)
            return_exp: Whether to return experiment object
            trial: Optuna trial (not used)
            save: Whether to save model
            use_early_stopping: Whether to use early stopping
        """
        trial_results = []
        self.aug_params['baseline_time'] = self.baseline_logger.total_time
        start_time = time.time()
        
        # Try different operation counts and magnitudes
        for ops in np.arange(1, self.aug_params['n_ops'] + 1):
            for mag in np.arange(self.aug_params['mag_bins']):
                sub_policy = []
                for op in range(ops):
                    sub_policy += ['uniform', mag]
                policy = [sub_policy]
                
                # Train with this policy and record results
                trial_logger = BasePolicy.fit_method(
                    self=self,
                    policy=policy,
                    return_exp=False,
                    use_early_stopping=self.args.early_stopping_aa
                )
                
                trial_info = {'policy': policy, 'val_loss': trial_logger.val_loss[-1]}
                trial_results.append(trial_info)
                torch.cuda.empty_cache()
                gc.collect()

        # Select and use best policy
        best_policy = self.select_best_policy(trial_results)
        self.aug_params['search_time'] = time.time() - start_time

        start_time = time.time()
        output = super().fit_method(
            best_policy, 
            return_exp, 
            save=self.args.save_aug, 
            use_early_stopping=self.args.early_stopping_aa
        )
        self.aug_params['final_time'] = time.time() - start_time
        return output


class TSAA(PolicySearch):
    """
    Automated Weighted Search for augmentation policy
    Uses Optuna for hyperparameter optimization
    """
    
    def __init__(self,
                max_top_k=3, 
                search='TPE',
                n_trials=100,
                random_state=None,
                aug_params=None,
                prune=True,
                start_random_ratio=0.3,
                reduction_factor=3, 
                augmix=False,   
                *args,
                **kwargs):
        """
        Initialize TSAA
        
        Args:
            max_top_k: Maximum number of top policies to select
            search: Search algorithm ('TPE' or 'Random')
            n_trials: Number of trials to run
            random_state: Random seed
            aug_params: Augmentation parameters
            prune: Whether to use pruning
            start_random_ratio: Ratio of initial random trials
            reduction_factor: Pruning reduction factor
            augmix: Whether to use AugMix-style augmentation
        """
        kwargs['method'] = 'TSAA'
        kwargs['max_top_k'] = max_top_k
        kwargs['aug_params'] = aug_params
        super().__init__(*args, **kwargs)
        
        if aug_params is None:
            self.aug_params['n_trials'] = n_trials
            self.aug_params['prune'] = prune
            self.aug_params['start_random_ratio'] = start_random_ratio 
            self.aug_params['search'] = search 
            self.aug_params['reduction_factor'] = reduction_factor
            self.aug_params['augmix'] = augmix
            self.aug_params['random_state'] = int(self.args.random_seed_aa) if random_state is None else int(random_state)
        else:
            if not all(i in aug_params for i in ['n_trials', 'search', 'random_state']):
                raise Exception('n_trials, search or random_state not defined')   
        
        self.study = None

    def fit_method(self, policy=None, return_exp=True, trial=None, save=False, use_early_stopping=False):
        """
        Run search to find best policies using Optuna
        
        Args:
            policy: Initial policy (not used)
            return_exp: Whether to return experiment object
            trial: Optuna trial (not used)
            save: Whether to save model
            use_early_stopping: Whether to use early stopping
        """
        self.aug_params['baseline_time'] = self.baseline_logger.total_time
        start_time = time.time()
        
        # Configure search algorithm
        if self.aug_params['search'] == 'TPE':
            n_startup_trials = int(self.aug_params['n_trials'] * self.aug_params['start_random_ratio'])
            sampler = optuna.samplers.TPESampler(
                seed=self.aug_params['random_state'],
                multivariate=True,
                group=True,
                n_startup_trials=n_startup_trials
            ) 
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.aug_params['random_state'])

        # Configure pruner
        pruner = optuna.pruners.NopPruner()
        if self.aug_params['prune']:
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                reduction_factor=self.aug_params['reduction_factor'], 
                min_resource=1
            )

        # Setup study and run optimization
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        self.study = optuna.create_study(sampler=sampler, pruner=pruner)
        self.study.optimize(self.objective, n_trials=self.aug_params['n_trials'])

        # Process results
        baseline_val_loss = self.baseline_logger.val_loss[-1]
        sorted_study_df = self.study.trials_dataframe().sort_values(by=['value'])
        
        # Take only policies better than baseline
        sorted_study_df = sorted_study_df[sorted_study_df['value'] < baseline_val_loss]

        # If no good policies found, return baseline
        if sorted_study_df.empty:
            if return_exp:
                return self.baseline_exp, self.baseline_logger
            else:
                return self.baseline_logger

        # Extract best policies
        sorted_trial_sub_policies = sorted_study_df.apply(self.construct_sub_policy, axis=1)
        best_policy = sorted_trial_sub_policies.values[:self.aug_params['max_top_k']].tolist()
        self.aug_params['search_time'] = time.time() - start_time
        
        # Train with best policy
        start_time = time.time()
        output = super().fit_method(
            best_policy, 
            return_exp, 
            save=self.args.save_aug, 
            use_early_stopping=self.args.early_stopping_aa
        )
        self.aug_params['final_time'] = time.time() - start_time
        return output

    def objective(self, trial):
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation loss
        """
        transformations_dict = AugmentTS(None, self.aug_params).transformations
        transformation_names = list(transformations_dict.keys())
        transformation_names_binary = [key for key, value in transformations_dict.items() if value["range"] is None]

        # Create sub-policy with sampled operations
        sub_policy = []
        for i in range(self.aug_params['n_ops']):
            if self.aug_params['augmix'] and i == 0:
                tr = trial.suggest_categorical(f'transformation_op_{i}', ['uniform'])
                mag = trial.suggest_float(f'{tr}_mag_op_{i}', 0, 1)
            else:
                tr = trial.suggest_categorical(f'transformation_op_{i}', transformation_names)
                if tr in transformation_names_binary:
                    # Binary transformations only take value 1.0
                    mag = trial.suggest_float(f'{tr}_mag_op_{i}', 1, 1, step=1)
                else:
                    eps = 0.0001  # Avoid 0 magnitude (equivalent to identity)
                    mag = trial.suggest_float(f'{tr}_mag_op_{i}', 0 + eps, 1)
            
            sub_policy += [tr, mag]
        
        policy = [sub_policy]
        
        # Train with policy and evaluate
        trial_logger = BasePolicy.fit_method(
            self=self,
            policy=policy,
            return_exp=False,
            trial=trial,
            use_early_stopping=False
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        val_loss = trial_logger.val_loss[-1]
        return val_loss

    def construct_sub_policy(self, x):
        """
        Convert Optuna trial parameters to a policy
        
        Args:
            x: Row from trials dataframe
            
        Returns:
            list: Constructed sub-policy
        """
        sub_policy = []
        for i in range(self.aug_params['n_ops']):
            op_i = f'_op_{i}'
            params = [item for item in x.index if op_i in item]
            params = [item for item in x[params] if str(item) != 'nan']
            op = np.flip(np.sort(params)).tolist()
            
            if len(op) != 2:
                raise Exception("Operation must be of length 2, transformation name and magnitude")
                
            op[1] = float(op[1])
            sub_policy += list(op)
        return sub_policy


class FastAA(PolicySearch):
    """
    Fast AutoAugment for time series
    Uses k-fold cross-validation to find policies efficiently
    """
    
    def __init__(self,
                max_top_k=2, 
                search='TPE',
                n_trials=100,
                random_state=None,
                aug_params=None,
                start_random_ratio=0.3, 
                augmix=False,
                *args,
                **kwargs):
        """
        Initialize FastAA
        
        Args:
            max_top_k: Maximum number of top policies to select
            search: Search algorithm ('TPE' or 'Random')
            n_trials: Number of trials to run
            random_state: Random seed
            aug_params: Augmentation parameters
            start_random_ratio: Ratio of initial random trials
            augmix: Whether to use AugMix-style augmentation
        """
        kwargs['method'] = 'FastAA'
        kwargs['max_top_k'] = max_top_k
        kwargs['aug_params'] = aug_params
        super().__init__(*args, **kwargs)
        
        if aug_params is None:
            self.aug_params['n_trials'] = n_trials
            self.aug_params['start_random_ratio'] = start_random_ratio 
            self.aug_params['search'] = search 
            self.aug_params['prune'] = False
            self.aug_params['augmix'] = augmix
            self.aug_params['random_state'] = int(self.args.random_seed_aa) if random_state is None else int(random_state)
        else:
            if not all(i in aug_params for i in ['n_trials', 'search', 'random_state']):
                raise Exception('n_trials, search or random_state not defined')   
        
        self.study = None

    def fit_method(self, policy=None, return_exp=True, trial=None, save=False, use_early_stopping=False):
        """
        Run k-fold search to find best policies
        
        Args:
            policy: Initial policy (not used)
            return_exp: Whether to return experiment object
            trial: Optuna trial (not used)
            save: Whether to save model
            use_early_stopping: Whether to use early stopping
        """
        best_policies = []
        
        # Run search on each fold
        for k in range(self.args.kfold):
            self.kfold_i = k
            self.baseline_k = Exp_Main(self.args)
            setting = self.setting + f'_kfold_{k}'
            self.baseline_k.train(setting, stage2=False, save=True, kfold=self.args.kfold, kfold_i=k)

            start_time = time.time()
            
            # Configure search algorithm
            if self.aug_params['search'] == 'TPE':
                n_startup_trials = int(self.aug_params['n_trials'] * self.aug_params['start_random_ratio'])
                sampler = optuna.samplers.TPESampler(
                    seed=self.aug_params['random_state'],
                    multivariate=True,
                    group=True,
                    n_startup_trials=n_startup_trials
                ) 
            else:
                sampler = optuna.samplers.RandomSampler(seed=self.aug_params['random_state'])

            # Configure pruner
            pruner = optuna.pruners.NopPruner()
            if self.aug_params['prune']:
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                    reduction_factor=self.aug_params['reduction_factor'],
                    min_resource=1
                )

            # Setup study and run optimization
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            self.study = optuna.create_study(sampler=sampler, pruner=pruner)
            self.study.optimize(self.objective, n_trials=self.aug_params['n_trials'])

            # Process results
            sorted_study_df = self.study.trials_dataframe().sort_values(by=['value'])
            sorted_trial_sub_policies = sorted_study_df.apply(self.construct_sub_policy, axis=1)
            best_policy = sorted_trial_sub_policies.values[:self.aug_params['max_top_k']].tolist()
            
            # Add to collection of best policies
            best_policies += best_policy
        
        # If no good policies found, return baseline
        if len(best_policies) == 0:
            if return_exp:
                return self.baseline_exp, self.baseline_logger
            else:
                return self.baseline_logger

        # Train with best policies
        start_time = time.time()
        output = super().fit_method(
            best_policies,
            return_exp,
            save=self.args.save_aug,
            use_early_stopping=self.args.early_stopping_aa
        )
        self.aug_params['final_time'] = time.time() - start_time
        return output

    def objective(self, trial):
        """
        Objective function for Optuna optimization using k-fold validation
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation loss
        """
        transformations_dict = AugmentTS(None, self.aug_params).transformations
        transformation_names = list(transformations_dict.keys())
        transformation_names_binary = [key for key, value in transformations_dict.items() if value["range"] is None]

        # Create sub-policy with sampled operations
        sub_policy = []
        for i in range(self.aug_params['n_ops']):
            if self.aug_params['augmix'] and i == 0:
                tr = trial.suggest_categorical(f'transformation_op_{i}', ['uniform'])
                mag = trial.suggest_float(f'{tr}_mag_op_{i}', 0, 1)
            else:
                tr = trial.suggest_categorical(f'transformation_op_{i}', transformation_names)
                if tr in transformation_names_binary:
                    # Binary transformations only take value 1.0
                    mag = trial.suggest_float(f'{tr}_mag_op_{i}', 1, 1, step=1)
                else:
                    eps = 0.0001  # Avoid 0 magnitude (equivalent to identity)
                    mag = trial.suggest_float(f'{tr}_mag_op_{i}', 0 + eps, 1)
            
            sub_policy += [tr, mag]
        
        policy = [sub_policy]

        # Fast evaluation using current fold
        print("Fit method, policy:", policy)
        augment_ts = AugmentTS(seasonality=self.args.seasonality, aug_params=self.aug_params)
        augment_ts.assign_policy(policy)
        self.baseline_k._set_ts_augment(augment_ts) 
        
        # Get validation data and evaluate
        vali_data, vali_loader = self.baseline_k._get_data(
            flag='train',
            kfold=self.args.kfold,
            kfold_i=self.kfold_i,
            kfold_flag='val'
        )
        val_loss = self.baseline_k.vali(vali_data, vali_loader, augment=True)
        print("val_loss fastaa: ", val_loss)
        return val_loss

    def construct_sub_policy(self, x):
        """
        Convert Optuna trial parameters to a policy
        
        Args:
            x: Row from trials dataframe
            
        Returns:
            list: Constructed sub-policy
        """
        sub_policy = []
        for i in range(self.aug_params['n_ops']):
            op_i = f'_op_{i}'
            params = [item for item in x.index if op_i in item]
            params = [item for item in x[params] if str(item) != 'nan']
            op = np.flip(np.sort(params)).tolist()
            
            if len(op) != 2:
                raise Exception("Operation must be of length 2, transformation name and magnitude")
                
            op[1] = float(op[1])
            sub_policy += list(op)
        return sub_policy

