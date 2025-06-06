import os
import json
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


NAMES = ['Perfect Bounds', 'Bad Bounds']
METHOD_NAMES = (
    # 'no shaping + clip w/o bound', 
    'Vanilla Q-UCB',
    'Q-UCB + Shaping', # 'shaping + clip w/ bound (ours)', 
    # 'no shaping + clip w/ bound', 
    # 'shaping + clip w/o bound'
)
ENV_NAMES = ['robot', 'windy']



fe = fm.FontEntry(
    fname=os.path.expanduser('~/Downloads/prima_serif_roman_bt.ttf'),
    name='primaserif')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
palette = sns.color_palette('Set3')

with open(f'data/{ENV_NAMES[0]}-{METHOD_NAMES[0]}-{NAMES[0]}-regret.json', 'r') as f:
    vanilla_q = json.load(f)
    vanilla_q = pd.DataFrame(data={'step':list(range(len(vanilla_q))), 'reg':vanilla_q})
# shift the x-axis a little bit to get the last element
ax = sns.lineplot(data=vanilla_q.iloc[::200000-1], x='step', y='reg', label=f'Vanilla Q-UCB', legend='brief', linewidth=1, color=palette[4], marker='o', markersize=8, mew=0)   

with open(f'data/{ENV_NAMES[0]}-{METHOD_NAMES[1]}-{NAMES[0]}-regret.json', 'r') as f:
    q_shaping_perfect = json.load(f)
    q_shaping_perfect = pd.DataFrame(data={'step':list(range(len(q_shaping_perfect))), 'reg':q_shaping_perfect})
ax = sns.lineplot(data=q_shaping_perfect.iloc[::200000-1], x='step', y='reg', label='Q-UCB Shaping + Upper Bounds', legend='brief', linewidth=1, color=palette[6], marker='*', markersize=9, mew=0)   

with open(f'data/{ENV_NAMES[0]}-{METHOD_NAMES[1]}-{NAMES[1]}-regret.json', 'r') as f:
    q_shaping_bad = json.load(f)
    q_shaping_bad = pd.DataFrame(data={'step':list(range(len(q_shaping_bad))), 'reg':q_shaping_bad})
ax = sns.lineplot(data=q_shaping_bad.iloc[::200000-1][:3], x='step', y='reg', label='Q-UCB Shaping + Lower Bounds', legend='brief', linewidth=1, color=palette[5], marker='o', markersize=8, mew=0)   

ax.set_xlabel('Environment Steps, 1e6', fontsize=8, labelpad=0, fontname='primaserif')
ax.set_ylabel('Cumu. Regrets, 1e5', fontsize=8, labelpad=0, fontname='primaserif')

ax.set_xticks(range(0, 2100000, 200000), labels=[str(round(i, ndigits=1)) for i in np.arange(0, 2.1, 0.2)], fontsize=10, fontname='primaserif')
# xmin, xmax = ax.get_xlim()
ax.set_yticks(range(0, 510000, 100000), labels=[str(i) for i in np.arange(0, 5.1, 1)], fontsize=10, fontname='primaserif')
plt.setp(ax.get_legend().get_texts(), fontsize='10', fontname='primaserif') 
ax.legend(bbox_to_anchor=(.95, .45), bbox_transform=ax.transAxes, prop=fm.FontProperties(family='primaserif'))
# ax.set_ylim(bottom=0.0, top=5.1)

sns.despine(offset=5)
plt.xticks(fontname='primaserif')
plt.yticks(fontname='primaserif')
sns.set_theme(font_scale=6)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(8, 6)
imgPath = f'figs/example_sample_efficiency.png'
fig.savefig(imgPath, dpi=800, bbox_inches='tight', pad_inches=0)
plt.close()
