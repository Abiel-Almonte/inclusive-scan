import os
import re
import subprocess
import optuna
import shutil
from jinja2 import Environment, FileSystemLoader

def generate_kernel(blockdim, unroll_factors, output_dir='generated', is_best: bool = False):
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('kernel_template.cu.j2')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not is_best:
        name = f"bd_{blockdim}-v_{unroll_factors['thread']}-s_{unroll_factors['serial']}-ld_{unroll_factors['load']}-st_{unroll_factors['store']}-va_{unroll_factors['vecs_add']}-iw_{unroll_factors['intra_warp']}-rw_{unroll_factors['inter_warp']}-wr_{unroll_factors['warp_reduce']}"
    else:
        name = "best"
    
    params = {
        'blockdim': blockdim,
        'thread_unroll_factor': unroll_factors['thread'] ,
        'load_unroll_factor': unroll_factors['load'],
        'store_unroll_factor': unroll_factors['store'],
        'serial_unroll_factor': unroll_factors['serial'],
        'intra_warp_unroll_factor': unroll_factors['intra_warp'],
        'inter_warp_unroll_factor': unroll_factors['inter_warp'],
        'warp_reduce_unroll_factor': unroll_factors['warp_reduce'],
        'vecs_add_unroll_factor': unroll_factors['vecs_add'],
    }

    rendered_template = template.render(params)
    file_path = os.path.join(output_dir, f"kernel_{name}.cuh")
    with open(file_path, 'w') as f:
        f.write(rendered_template)
    
    return os.path.basename(file_path)

def objective(trial):
    blockdim = trial.suggest_categorical('blockdim', [32, 64, 128, 256, 512, 1024])
    unroll_factors = {
        'thread': trial.suggest_categorical('thread_unroll_factor', [1, 2, 4, 8]),
        'load': trial.suggest_categorical('load_unroll_factor', [0, 1, 2, 4, 8]),
        'store': trial.suggest_categorical('store_unroll_factor', [0, 1, 2, 4, 8]),
        'serial': trial.suggest_categorical('serial_unroll_factor', [0, 1, 2, 4, 8]),
        'intra_warp': trial.suggest_categorical('intra_warp_unroll_factor', [0, 1, 4, 8, 16, 32]),
        'inter_warp': trial.suggest_categorical('inter_warp_unroll_factor', [0, 1, 4, 8, 16, 32]),
        'warp_reduce': trial.suggest_categorical('warp_reduce_unroll_factor', [0, 1, 4, 8, 16, 32]),
        'vecs_add': trial.suggest_categorical('vecs_add_unroll_factor', [0, 1, 2, 4, 8]),
    }

    thread_unroll_size = unroll_factors['thread']
    for key in ['load', 'store', 'serial', 'vecs_add']:
        if unroll_factors[key] > thread_unroll_size:
            raise optuna.exceptions.TrialPruned()


    output_dir = 'generated'
    try:
        kernel_filename = generate_kernel(blockdim, unroll_factors, output_dir=output_dir)
    except Exception as e:
        print(f"Trial {trial.number}: Kernel generation failed: {e}")
        
        return 0.0

    kernel_path = os.path.join(output_dir, kernel_filename)
    
    try:
        result = subprocess.run(
            ['./build_and_run.sh', 'benchmark', kernel_path],
            check=True, capture_output=True, text=True, timeout=60
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if isinstance(e, subprocess.CalledProcessError):
             print(f"Stderr: {e.stderr}")
        return 0.0

    output = result.stdout
    match = re.search(r"My Kernel Performance: ([\d.]+) GB/s", output)
    if not match:
        return 0.0
        
    bandwidth = float(match.group(1))
    
    return bandwidth

if __name__ == "__main__":
    storage_name = "sqlite:///db.sqlite3"
    study_name = "cuda_kernel_tuning"
    study = optuna.create_study(
        direction='maximize', 
        storage=storage_name, 
        study_name=study_name, 
        load_if_exists=True
    )
    
    try:
        study.optimize(objective, n_trials=500, show_progress_bar=True)
    finally:
        if os.path.exists('generated'):
            shutil.rmtree('generated')
    
    best_trial = study.best_trial
    
    best_params = best_trial.params
    best_unroll_factors = {
        'thread': best_params['thread_unroll_factor'] ,
        'load': best_params['load_unroll_factor'],
        'store': best_params['store_unroll_factor'],
        'serial': best_params['serial_unroll_factor'],
        'intra_warp': best_params['intra_warp_unroll_factor'],
        'inter_warp': best_params['inter_warp_unroll_factor'],
        'warp_reduce': best_params['warp_reduce_unroll_factor'],
        'vecs_add': best_params['vecs_add_unroll_factor'],
    }
    
    best_kernel_filename = generate_kernel(
        best_params['blockdim'],
        best_unroll_factors,
        output_dir='.',
        is_best = True,
    )
    print(f"\nBest kernel saved to best_kernel/{best_kernel_filename}")
    print(f"Bandwidth: {best_trial.value:.2f} GB/s")
    print("Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}") 