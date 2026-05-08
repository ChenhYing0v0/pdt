import torch
import os
import sys
import numpy as np
import argparse
from thop import profile

# Mock pandas to avoid import errors in utils.tools if not installed
try:
    import pandas
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["pandas"] = MagicMock()

# Mock matplotlib to avoid import errors in utils.tools if not installed/headless
try:
    import matplotlib.pyplot as plt
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.pyplot"] = MagicMock()

# Mock reformer_pytorch to avoid import errors
try:
    import reformer_pytorch
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules["reformer_pytorch"] = MagicMock()

from models import iTransformer, DLinear, PatchTST, TimesNet, Crossformer, R2Linear

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Configurations (Basic info like enc_in, etc.)
DATASET_INFO = {
    'weather': {
        'enc_in': 21,
        'dec_in': 21,
        'c_out': 21,
        'pl_list': [96, 192, 336, 720],
        'root_path': './dataset/weather/',
        'data_path': 'weather.csv',
        'r_rank_list': [96, 96, 96, 96],
        'k_top_list': [16, 16, 16, 16],
        'alpha_list': [0.1, 0.4, 0.25, 0.05],
        'thr_list': [0.05, 0.1, 0.25, 0.05],
        'Q_chan_indep': 0,
        'mat_path_pattern': './dataset/RRR_mats/weather/weather_RRR_L96_R{r_rank}_H{pl}_{type}.npy',
        'q_out_mat_pattern': './dataset/PCCA_mats/weather/weather_PCCA_OUT_H{pl}_identity_avg.npy'
    },
    'ECL': {
        'enc_in': 321,
        'dec_in': 321,
        'c_out': 321,
        'pl_list': [96, 192, 336, 720],
        'root_path': './dataset/electricity/',
        'data_path': 'electricity.csv',
        'r_rank_list': [96, 96, 96, 96],
        'k_top_list': [16, 16, 16, 16],
        'alpha_list': [0.05, 0.1, 0.1, 0.1],
        'thr_list': [0.25, 0.05, 0.1, 0.1],
        'Q_chan_indep': 0,
        'mat_path_pattern': './dataset/RRR_mats/electricity/electricity_RRR_L96_R{r_rank}_H{pl}_{type}.npy',
        'q_out_mat_pattern': './dataset/PCCA_mats/electricity/electricity_PCCA_OUT_H{pl}_identity_avg.npy'
    },
    'ETTh1': {
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'pl_list': [96, 192, 336, 720],
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'r_rank_list': [96, 96, 96, 96],
        'k_top_list': [16, 16, 16, 16],
        'alpha_list': [0.15, 0.05, 0.05, 0.5],
        'thr_list': [0.05, 0.05, 0.05, 0.05],
        'Q_chan_indep': 0,
        'mat_path_pattern': './dataset/RRR_mats/ETTh1/ETTh1_RRR_L96_R{r_rank}_H{pl}_{type}.npy',
        'q_out_mat_pattern': './dataset/PCCA_mats/ETTh1/ETTh1_PCCA_OUT_H{pl}_identity_avg.npy'
    },
    'ETTh2': {
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'pl_list': [96, 192, 336, 720],
        # Assuming ETTh2 uses similar config to ETTh1 for R2Linear if needed, or just placeholders
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'r_rank_list': [96, 96, 96, 96],
        'k_top_list': [16, 16, 16, 16],
        'alpha_list': [0.15, 0.05, 0.05, 0.5],
        'thr_list': [0.05, 0.05, 0.05, 0.05],
        'Q_chan_indep': 0,
        'mat_path_pattern': './dataset/RRR_mats/ETTh2/ETTh2_RRR_L96_R{r_rank}_H{pl}_{type}.npy',
        'q_out_mat_pattern': './dataset/PCCA_mats/ETTh2/ETTh2_PCCA_OUT_H{pl}_identity_avg.npy'
    }
}

# Model Specific Configurations per Dataset
# If a (model, dataset) pair is missing, it falls back to 'default' or global defaults.
MODEL_CONFIGS = {
    ('iTransformer', 'weather'): {
        'e_layers': 3, 'd_layers': 1, 'factor': 3, 'd_model': 512, 'd_ff': 512
    },
    ('iTransformer', 'ETTh2'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 128, 'd_ff': 128
    },
    # Assuming ETTh1 is similar to ETTh2 for iTransformer if not explicitly found
    ('iTransformer', 'ETTh1'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 128, 'd_ff': 128
    },

    ('PatchTST', 'weather'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'n_heads': 4, 'd_model': 512, 'd_ff': 2048
    },
    ('PatchTST', 'ETTh1'): {
        'e_layers': 1, 'd_layers': 1, 'factor': 3, 'n_heads': 8, 'd_model': 512, 'd_ff': 2048
    },

    ('TimesNet', 'weather'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 32, 'd_ff': 32, 'top_k': 5
    },
    ('TimesNet', 'ETTh1'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 16, 'd_ff': 32, 'top_k': 5
    },

    ('Crossformer', 'weather'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 32, 'd_ff': 32
    },
    ('Crossformer', 'ETTh1'): {
        'e_layers': 2, 'd_layers': 1, 'factor': 3, 'd_model': 512, 'd_ff': 2048
    },

    ('DLinear', 'ETTh1'): {
        'e_layers': 2, 'dropout': 0.2
    },
    # DLinear usually doesn't change much, but we can add defaults if needed
    
    ('R2Linear', 'weather'): {
        'embed_size': 16, 'dropout': 0.0, 'activation': 'gelu', 'CKA_flag': 0, 'freeze_R': 0, 'mask_sharpness_k': 2, 'temp_patch_len': 16, 'temp_stride': 8
    },
    ('R2Linear', 'ECL'): {
        'embed_size': 16, 'dropout': 0.0, 'activation': 'gelu', 'CKA_flag': 0, 'freeze_R': 0, 'mask_sharpness_k': 2, 'temp_patch_len': 16, 'temp_stride': 8
    },
    ('R2Linear', 'ETTh1'): {
        'embed_size': 16, 'dropout': 0.0, 'activation': 'gelu', 'CKA_flag': 0, 'freeze_R': 0, 'mask_sharpness_k': 2, 'temp_patch_len': 16, 'temp_stride': 8
    },
    ('R2Linear', 'ETTh2'): {
        'embed_size': 16, 'dropout': 0.0, 'activation': 'gelu', 'CKA_flag': 0, 'freeze_R': 0, 'mask_sharpness_k': 2, 'temp_patch_len': 16, 'temp_stride': 8
    }
}

class Config:
    def __init__(self, dataset_name, pl, model_name):
        self.task_name = 'long_term_forecast'
        self.is_training = 0
        self.model_id = 'test'
        self.model = model_name
        self.data = 'custom'
        self.features = 'M'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = pl
        
        # Dataset info
        ds_info = DATASET_INFO.get(dataset_name, DATASET_INFO['weather'])
        self.enc_in = ds_info['enc_in']
        self.dec_in = ds_info['dec_in']
        self.c_out = ds_info['c_out']
        self.root_path = ds_info.get('root_path', './dataset/weather/')
        self.data_path = ds_info.get('data_path', 'weather.csv')
        
        # R2Linear specific dataset params
        if model_name == 'R2Linear':
            try:
                idx = ds_info['pl_list'].index(pl)
                self.r_rank = ds_info['r_rank_list'][idx]
                self.k_top = ds_info['k_top_list'][idx]
                self.alpha_init = ds_info['alpha_list'][idx]
                self.mask_threshold = ds_info['thr_list'][idx]
                self.Q_chan_indep = ds_info['Q_chan_indep']
                
                r_rank = self.r_rank
                k_top = self.k_top
                
                self.Q_MAT_file = ds_info['mat_path_pattern'].format(r_rank=r_rank, pl=pl, type='Qin')
                self.q_mat_file = self.Q_MAT_file
                
                self.R_MAT_file = ds_info['mat_path_pattern'].format(r_rank=r_rank, pl=pl, type='R')
                self.r_mat_file = self.R_MAT_file
                
                self.Rk_MAT_file = ds_info['mat_path_pattern'].format(r_rank=k_top, pl=pl, type='R')
                self.rk_mat_file = self.Rk_MAT_file
                
                self.Q_OUT_MAT_file = ds_info['q_out_mat_pattern'].format(pl=pl)
                self.q_out_mat_file = self.Q_OUT_MAT_file
            except (ValueError, KeyError) as e:
                print(f"Warning: Missing R2Linear config for {dataset_name} pl={pl}. Using defaults/placeholders. Error: {e}")
                self.r_rank = 96
                self.k_top = 96
                self.alpha_init = 0.1
                self.mask_threshold = 0.05
                self.Q_chan_indep = 0
                self.Q_MAT_file = "dummy_Q.npy"
                self.q_mat_file = "dummy_Q.npy"
                self.R_MAT_file = "dummy_R.npy"
                self.r_mat_file = "dummy_R.npy"
                self.Rk_MAT_file = "dummy_Rk.npy"
                self.rk_mat_file = "dummy_Rk.npy"
                self.Q_OUT_MAT_file = "dummy_Qout.npy"
                self.q_out_mat_file = "dummy_Qout.npy"

        
        # Default Model Params (from run.py)
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.factor = 1
        self.top_k = 5
        self.num_kernels = 6
        self.moving_avg = 25
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.channel_independence = 0
        
        # Missing attributes found during first run
        self.freq = 'h'
        self.individual = False
        
        # Apply Model Specific Overrides
        if (model_name, dataset_name) in MODEL_CONFIGS:
            overrides = MODEL_CONFIGS[(model_name, dataset_name)]
            for k, v in overrides.items():
                setattr(self, k, v)
        
        # Extra params that might be needed by some models
        self.seg_len = 48 # Crossformer default?
        self.win_size = 1 # Crossformer default?
        self.cross_activation = 'tanh' # Crossformer default?
        
        # PatchTST specific
        self.fc_dropout = 0.05
        self.head_dropout = 0.0
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.revin = 1
        self.affine = 0
        self.subtract_last = 0
        self.decomposition = 0
        self.kernel_size = 25
        
        # iTransformer specific
        self.use_norm = 1
        self.class_strategy = 'projection'

def measure_training_memory(model_class, config, model_name, batch_size=32):
    try:
        # Set training mode
        config.is_training = 1
        model = model_class.Model(config).to(device)
        model.train()
        
        # Optimizer (Adam is standard)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Dummy inputs
        dummy_input = torch.randn(batch_size, config.seq_len, config.enc_in).to(device)
        dummy_input_mark = torch.randn(batch_size, config.seq_len, 4).to(device)
        dummy_dec_in = torch.randn(batch_size, config.pred_len + config.label_len, config.dec_in).to(device)
        dummy_dec_mark = torch.randn(batch_size, config.pred_len + config.label_len, 4).to(device)
        dummy_target = torch.randn(batch_size, config.pred_len, config.c_out).to(device)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # Training step
        optimizer.zero_grad()
        outputs = model(dummy_input, dummy_input_mark, dummy_dec_in, dummy_dec_mark)
        
        # Handle tuple output if any
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # Calculate loss
        # Ensure output shape matches target. 
        # Some models might output [B, L, D]
        # R2Linear returns [B, L, D]
        loss = criterion(outputs, dummy_target)
        
        loss.backward()
        optimizer.step()
        
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated()
            
        return max_memory
    except Exception as e:
        print(f"Error measuring training mem for {model_name}: {e}")
        # import traceback
        # traceback.print_exc()
        return None

def measure_model(model_class, config, model_name):
    try:
        model = model_class.Model(config).to(device)
        model.eval()
        
        dummy_input = torch.randn(1, config.seq_len, config.enc_in).to(device)
        
        # Measure MACs and Params
        # Note: Some models might require different inputs or handling
        # For example, Crossformer might need x_mark (time features)
        # We'll create dummy time features just in case
        dummy_input_mark = torch.randn(1, config.seq_len, 4).to(device) # 4 is typical for timeF
        dummy_dec_in = torch.randn(1, config.pred_len + config.label_len, config.dec_in).to(device)
        dummy_dec_mark = torch.randn(1, config.pred_len + config.label_len, 4).to(device)
        
        # Most models in this repo take (x_enc, x_mark_enc, x_dec, x_mark_dec)
        # But thop.profile expects a single input or a tuple of inputs.
        # We need to wrap the model call to handle the arguments if thop doesn't support kwargs well.
        
        # However, many of these models' forward method signature is:
        # forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        
        # Let's try to pass all 4 arguments.
        inputs = (dummy_input, dummy_input_mark, dummy_dec_in, dummy_dec_mark)
        
        macs, params = profile(model, inputs=inputs, verbose=False)
        
        # Measure GPU Memory
        max_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            with torch.no_grad():
                _ = model(*inputs)
            max_memory = torch.cuda.max_memory_allocated()
            
        return macs, params, max_memory
    except Exception as e:
        print(f"Error measuring {model_name}: {e}")
        # import traceback
        # traceback.print_exc()
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weather', help='Dataset name: weather, ECL, ETTh1')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    if dataset_name not in DATASET_INFO:
        print(f"Dataset {dataset_name} not found in config.")
        return

    print(f"Measuring resources for dataset: {dataset_name}")
    print(f"{'Pred Len':<10} | {'Model':<15} | {'MACs (G)':<10} | {'Params (M)':<10} | {'Inf Mem (MB)':<12} | {'Train Mem (MB)':<14}")
    print("-" * 90)

    pl_list = DATASET_INFO[dataset_name]['pl_list']
    
    models_to_measure = [
        ('iTransformer', iTransformer),
        ('DLinear', DLinear),
        ('PatchTST', PatchTST),
        ('TimesNet', TimesNet),
        ('Crossformer', Crossformer),
        ('R2Linear', R2Linear)
    ]

    for pl in pl_list:
        for model_name, model_class in models_to_measure:
            config = Config(dataset_name, pl, model_name)
            
            # Measure Inference
            macs, params, inf_mem = measure_model(model_class, config, model_name)
            
            # Measure Training
            train_mem = measure_training_memory(model_class, config, model_name)
            
            if macs is not None:
                train_mem_str = f"{train_mem/1024/1024:<14.2f}" if train_mem is not None else "Failed"
                print(f"{pl:<10} | {model_name:<15} | {macs/1e9:<10.4f} | {params/1e6:<10.4f} | {inf_mem/1024/1024:<12.2f} | {train_mem_str}")
            else:
                print(f"{pl:<10} | {model_name:<15} | {'Failed':<10} | {'Failed':<10} | {'Failed':<12} | {'Failed':<14}")
        
        print("-" * 90)

if __name__ == "__main__":
    main()
