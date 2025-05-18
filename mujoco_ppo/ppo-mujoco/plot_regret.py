import pandas as pd, matplotlib.pyplot as plt
df_a = pd.read_csv('results/brute/summary.csv')
df_b = pd.read_csv('results/causal/summary.csv')
plt.scatter(df_a['mask_size'], df_a['cum_regret'], label='2^N sweep', alpha=.4)
plt.scatter(df_b['mask_size'], df_b['cum_regret'], label='causal subsets', marker='^')
plt.xlabel('Number of free DoF')
plt.ylabel('Cumulative Regret')
plt.legend(); plt.grid()
plt.savefig('compare_regret.png', dpi=300)