"""Print a per-epoch checkpoint table across all experiments."""

import json

names = ['baseline', 'altflip', 'whitening', 'dirac',
         'scalebias_lookahead', 'all_methods']
dirs = [f'experiments/{n}/outputs' for n in names]

header = '{:>6}'.format('Epoch')
for n in names:
    header += '{:>22}'.format(n)
print(header)
print('-' * 140)

data = [json.load(open(d + '/results.json')) for d in dirs]
for ep in [1, 5, 10, 20, 30, 50, 75, 100]:
    line = '{:>6}'.format(ep)
    for d in data:
        row = next((r for r in d['log'] if r['epoch'] == ep), None)
        v = '{:.1f}%'.format(row['val_acc'] * 100) if row else '-'
        line += '{:>22}'.format(v)
    print(line)

print()
print('Final summary (best val ckpt):')
fmt = '{:<22} {:>11} {:>9} {:>10}'
print(fmt.format('Experiment', 'Best Val', 'Test', 'Time(m)'))
print('-' * 55)
for n, d in zip(names, data):
    print(fmt.format(n,
                     '{:.2f}%'.format(d['best_val_acc'] * 100),
                     '{:.2f}%'.format(d['test_acc'] * 100),
                     '{:.1f}'.format(d['total_time_sec'] / 60)))
