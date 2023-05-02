import os
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Times New Roman',
        'size' : 12}

matplotlib.rc('font', **font)

# Search for TensorBoard log directories recursively
root_dir = 'logs/'
log_dirs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    if any(filename.startswith('events.') for filename in filenames):
        log_dirs.append(dirpath)

# Load scalar data from all log directories
scalar_data_er = []
scalar_data_cu = []

for log_dir in log_dirs:
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']

    for tag in tags:
        if tag.endswith('episodic_reward'):
            data = [(scalar.wall_time, scalar.step, scalar.value, log_dir.split('/')[-1]) for scalar in event_acc.Scalars(tag)]
            scalar_data_er.extend(data)
        
        if tag.endswith('connected_users'):
            data = [(scalar.wall_time, scalar.step, scalar.value, log_dir.split('/')[-1]) for scalar in event_acc.Scalars(tag)]
            scalar_data_cu.extend(data)

# Convert scalar data to pandas DataFrame
df_er = pd.DataFrame(scalar_data_er, columns=['Time', 'Step', 'Episodic Reward', 'Log'])

df_cu = pd.DataFrame(scalar_data_cu, columns=['Time', 'Step', 'Connected Users', 'Log'])

df_er = df_er[df_er.Log != "logs\\"]
df_cu = df_cu[df_er.Log != "logs\\"]

# Smooth the scalar data using a rolling mean
df_er['Smoothed Reward'] = df_er.groupby('Log')['Episodic Reward'].rolling(window=160, min_periods=1, closed='right').mean().values
df_cu['Smoothed Users'] = df_cu.groupby('Log')['Connected Users'].rolling(window=160, min_periods=1, closed='right').mean().values

# labels = ["Distance Threshold = 1000", "Distance Threshold = 0", "Distance Threshold = 250", "Distance Threshold = 500", "Distance Threshold = 750"]
# labels = ["5 UAVs", "7 UAVs"]
# labels = ["Level 1: Implicit Exchange", "Level 2: Reward Exchange", "Level 3: Position Exchange"]
labels = ["MAQL", "MADQL"]
logs_labels = pd.unique(df_cu.Log)
labels_dict = dict(zip(logs_labels, labels))

# del labels_dict["madql_uav__lvl2__5__1__1682907964"]
# labels_dict["madql_uav__lvl2__5__1__1682907964"] = "Distance Threshold = 1000"

palette = sns.color_palette('Set1', 2)
# Plot the smoothed scalar data using Seaborn
sns.set_style('whitegrid')
ax = sns.lineplot(x='Step', y='Smoothed Reward', hue='Log', hue_order= labels_dict.keys(), data=df_er, legend=True, palette=palette, linewidth=1.5)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles[:5], labels_dict.values())
ax = sns.lineplot(x='Step', y='Episodic Reward', hue='Log', hue_order= labels_dict.keys(), data=df_er, legend=False, palette=palette, alpha = 0.2)
ax.set_title('Convergence Plot: MAQL vs MADQL with Position Exchange')
ax.set_xlabel('Episode', fontsize = 14, family = 'Times New Roman')
ax.set_ylabel('Episodic Reward', fontsize = 14, family = 'Times New Roman')
plt.show()

sns.set_style('whitegrid')
ax = sns.lineplot(x='Step', y='Smoothed Users', hue='Log', hue_order= labels_dict.keys(), data=df_cu, legend=True, palette=palette,linewidth=1.5)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles[:5], labels_dict.values())
ax = sns.lineplot(x='Step', y='Connected Users', hue='Log', hue_order= labels_dict.keys(), data=df_cu, legend=False, palette=palette, alpha = 0.2)
ax.set_title('Convergence with Connected Users: MAQL vs MADQL with Position Exchange')
ax.set_xlabel('Episode', fontsize = 14, family = 'Times New Roman')
ax.set_ylabel('Connected Users', fontsize = 14, family = 'Times New Roman')
plt.show()




