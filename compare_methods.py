import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

methods    = ['Direct\n(No Avoidance)', 'Reactive\nOnly', 'A*+Reactive\n(Classical)', 'PPO\n(RL)']
success    = [0,    88,   100,  100 ]
collisions = [60,   7,    0,    2   ]
path_len   = [0,    5.58, 4.95, 16.16]
deliv_time = [0,    18.3, 16.8, 6.7 ]
colors     = ['#e05050','#f0a030','#4a7fe0','#4caf50']

fig, axs = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Multi-UAV Delivery — Full Method Comparison\n(Classical vs Reinforcement Learning)',
             fontsize=14, fontweight='bold')

def bar(ax, vals, title, ylabel, fmt='{:.0f}', ymax=None):
    bars = ax.bar(methods, vals, color=colors, width=0.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    if ymax: ax.set_ylim(0, ymax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                fmt.format(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

bar(axs[0], success,    'Success Rate (%)',    '%',      '{:.0f}', ymax=115)
bar(axs[1], collisions, 'Total Collisions',    'count',  '{:.0f}')
bar(axs[2], path_len,   'Avg Path Length (m)', 'meters', '{:.2f}')
bar(axs[3], deliv_time, 'Delivery Time (s)',   'seconds','{:.1f}')

plt.tight_layout()
out = '/Users/nah/Desktop/method_comparison.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print('Saved', out)
subprocess.run(['open', out])
