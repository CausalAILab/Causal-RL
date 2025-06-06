import os
import json
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

from constants import ENV_NAMES, MAX_EPISODES, SEEDS


fe = fm.FontEntry(
    fname=os.path.expanduser('~/Downloads/prima_serif_roman_bt.ttf'),
    name='primaserif')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
palette = sns.color_palette('Set3')

METHOD_NAMES = ['Vanilla Q-UCB', 'Shaping + Min Beh. Value', 'Shaping + Max Beh. Value', 'Shaping + Avg Beh. Value', 'Shaping + BCQ', 'Shaping + Causal Bound (Ours)']
COLORS = [palette[0], palette[2], palette[8], palette[4], palette[9], palette[5]]

# for env_name in ENV_NAMES:
for env_name in ['MiniGrid-Empty-8x8-v0', 'Custom-LavaCrossing-easy-v0', 'Custom-LavaCrossing-extreme-v0', 'Custom-LavaCrossing-maze-complex-v0']:
    print(env_name)
    sns.reset_defaults()
    stepsize = MAX_EPISODES[env_name]//8
    max_y = 0
    for i, method in enumerate(METHOD_NAMES):
        full_regret = []
        full_seeds = []
        full_steps = []
        for seed in SEEDS:
            with open(f'regrets/REG-{env_name}-{method}-{seed}.json', 'r') as f:
                regret = json.load(f)
                # max_y = max(max(regret), max_y)
                max_y = max(max(np.log(regret)), max_y)
                regret[0] = 1 #for better visuals in log scale
                full_regret += regret[::stepsize-1]
                full_seeds += [seed,]*len(regret[::stepsize-1])
                full_steps += list(range(len(regret)))[::stepsize-1]
        # regret = pd.DataFrame(data={'step':full_steps, 'reg':full_regret, 'seed':full_seeds}).sort_values('step', ascending=True)
        regret = pd.DataFrame(data={'step':full_steps, 'reg':np.log(full_regret), 'seed':full_seeds}).sort_values('step', ascending=True)
        # shift the x-axis a little bit to get the last element
        marker = 'o' if method != 'Shaping + Causal Bound (Ours)' else '*'
        markersize = 6 if method != 'Shaping + Causal Bound (Ours)' else 10
        alpha = 1.0 if method != 'Shaping + Causal Bound (Ours)' else 1.0
        ax = sns.lineplot(data=regret, x='step', y='reg', label=f'{method}', legend='brief', linewidth=2, color=COLORS[i], marker=marker, markersize=markersize, mew=0, alpha=alpha)   


    ax.set_xlabel('Environment Steps, 1e'+str(len(str(MAX_EPISODES[env_name])[1:])), fontsize=10, labelpad=0, fontname='primaserif')
    ax.set_ylabel('Cumu. Regrets, log-scale', fontsize=10, labelpad=0, fontname='primaserif')
    
    magnitude = 10**int(f'{MAX_EPISODES[env_name]:.1E}'.split('+')[1])
    ax.set_xticks(range(0, MAX_EPISODES[env_name]+1, stepsize), labels=[f'{i/magnitude:.1f}' for i in range(0, MAX_EPISODES[env_name]+1, stepsize)], fontsize=12, fontname='primaserif')
    # ax.set_xticks(range(0, MAX_EPISODES[env_name]+1, stepsize), labels=[f'{i:.1E}'.split('E')[0] for i in range(0, MAX_EPISODES[env_name]+1, stepsize)], fontsize=10, fontname='primaserif')
    # xmin, xmax = ax.get_xlim()
    ax.set_yticks(range(0, int(np.ceil(max_y))+1, 2), labels=[str(i) for i in np.arange(0, int(np.ceil(max_y))+1, 2)], fontsize=12, fontname='primaserif')
    plt.setp(ax.get_legend().get_texts(), fontsize='10', fontname='primaserif') 
    # Calculate maximum label length
    max_label_length = max(len(label.get_text()) for label in plt.legend().get_texts())
    # Adjust font size based on label length
    fontsize = 11 - max_label_length * 0.2
    plt.legend(fontsize=fontsize)
    # ax.legend(bbox_to_anchor=(.95, .45), bbox_transform=ax.transAxes, prop=fm.FontProperties(family='primaserif'))
    # ax.set_ylim(bottom=0.0, top=5.1)

    sns.despine(offset=5)
    plt.xticks(fontname='primaserif')
    plt.yticks(fontname='primaserif')
    sns.set_theme(font_scale=8)
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    imgPath = f'figs/exp-reg-{env_name}.png'
    fig.savefig(imgPath, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # break

