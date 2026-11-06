import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta

models = {
    'R=4': 'outputs/gen_ar_bbh_r4.pt',
    'R=8': 'outputs/gen_ar_bbh_r8.pt',
    'R=16 (K=4)': 'outputs/gen_ar_bbh_r16_k4.pt',
}
seeds = [12345, 12346, 12347]
depths = [1, 2, 4, 8]

print('\n' + '=' * 60)
print(' BBH Compression Grid: Multi-Seed Evaluation (N=2048 x 3 Seeds)')
print('=' * 60)

with open('outputs/bbh_multiseed_sweep_results.txt', 'w') as f:
    f.write('=' * 60 + '\n')
    f.write(' BBH Compression Grid: Multi-Seed Evaluation (N=2048 x 3 Seeds)\n')
    f.write('=' * 60 + '\n')
    
    for name, path in models.items():
        print(f'\n--- Model: {name} ---')
        f.write(f'\n--- Model: {name} ---\n')
        gen, ae, cond_ae, meta = load_generator(path, device='cpu')
        task = _task_from_meta(meta)
        
        depth_records = {d: [] for d in depths}
        for s in seeds:
            res = eval_concept_generator(gen, ae, task, depths=depths, n_per_depth=2048, seed=s, device='cpu')
            for d in depths:
                depth_records[d].append(res[d] * 100)
                
        for d in depths:
            mean = np.mean(depth_records[d])
            std = np.std(depth_records[d])
            line = f'Depth {d}: {mean:6.2f}% +/- {std:4.2f}%'
            print(line)
            f.write(line + '\n')

    f.write('=' * 60 + '\n')

print('\n' + '=' * 60)
print('Results successfully saved to outputs/bbh_multiseed_sweep_results.txt')
