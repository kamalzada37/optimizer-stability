# analyze.py
import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir='../results'):
    files = glob.glob(os.path.join(results_dir, 'res_*.json'))
    rows = []
    for f in files:
        r = json.load(open(f))
        m = r['meta']
        final_acc = r['history']['test_acc'][-1] if len(r['history']['test_acc'])>0 else None
        rows.append({
            'file': f,
            'optimizer': m['optimizer'],
            'lr': m['lr'],
            'noise': m['noise'],
            'precision': m['precision'],
            'seed': m['seed'],
            'final_acc': final_acc,
            'diverged': r.get('diverged', False),
            'elapsed_sec': r.get('elapsed_sec', None)
        })
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = load_results('../results')
    if df.empty:
        print("No results found in ../results. Run experiments first.")
    else:
        print(df.groupby(['optimizer','noise','precision'])['final_acc'].agg(['mean','std','count']))
        for noise in sorted(df['noise'].unique()):
            sub = df[df['noise']==noise]
            pivot = sub.pivot_table(index='optimizer', columns='precision', values='final_acc', aggfunc=['mean','std'])
            print("Noise:",noise)
            print(pivot)
            try:
                mean = pivot['mean']
                err = pivot['std']
                ax = mean.plot.bar(yerr=err, rot=0, title=f'Noise {noise}')
                plt.tight_layout()
                plt.savefig(f'../results/plot_noise{int(100*noise)}.png')
                plt.close()
            except Exception as e:
                print("Plotting skipped:", e)
