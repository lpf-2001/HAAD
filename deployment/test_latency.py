import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# ------------------- 统计函数 -------------------
def process_test_for_latency(file_paths):
    """提取有效时延数据"""
    all_data = []
    for path in file_paths:
        df = pd.read_csv(path)
        valid_data = df.loc[df['http_code']==200, 'curl_time_total'].dropna().tolist()
        all_data.extend(valid_data)
    return all_data

# ------------------- 数据处理 -------------------
groups = {
    '5 Patches (30 packets)': glob.glob('./时延测试/1/test*/tor_test_results.csv'),
    '6 Patches (35 packets)': glob.glob('./时延测试/2/test*/tor_test_results.csv'),
    '7 Patches (42 packets)': glob.glob('./时延测试/3/test*/tor_test_results.csv'),
    '8 Patches (45 packets)': glob.glob('./时延测试/4/test*/tor_test_results.csv'),
    'Normal Forward': glob.glob('./时延测试/NoPert/test*/tor_test_results.csv')
}

# 统计 20%-80% 分位数
stats = {}
for group_name, file_list in groups.items():
    data = process_test_for_latency(file_list)
    if len(data) > 0:
        stats[group_name] = {
            'p20': np.percentile(data, 20),
            'p80': np.percentile(data, 80),
            'median': np.median(data)
        }
    else:
        stats[group_name] = {'p20': 0, 'p80': 0, 'median': 0}

# ------------------- 绘制图表 -------------------
labels = list(stats.keys())
x = np.arange(len(labels))
bar_width = 0.45

p20 = [stats[g]['p20'] for g in labels]
p80 = [stats[g]['p80'] for g in labels]
median = [stats[g]['median'] for g in labels]
print(median)

# y 轴范围
y_min = min(p20) * 0.9
y_max = max(p80) * 1.1
y_range = y_max - y_min

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2

# 绘制20%-80%区间（矩形条）
for i, lbl in enumerate(labels):
    plt.fill_between(
        [x[i] - bar_width/2, x[i] + bar_width/2],
        p20[i], p80[i],
        color='#76B7B2', alpha=0.8,
        edgecolor='#333333', linewidth=1.2,
        zorder=3
    )
    # 画中位线
    plt.plot([x[i] - bar_width/2, x[i] + bar_width/2],
             [median[i], median[i]],
             color='#1B4F72', linewidth=2.2, zorder=4)
    # 标注数值
    plt.text(x[i], p80[i] + 0.015*y_range, f"{p80[i]:.2f}",
             ha='center', va='bottom', fontsize=12, color='#333333')
    plt.text(x[i], p20[i] - 0.015*y_range, f"{p20[i]:.2f}",
             ha='center', va='top', fontsize=12, color='#333333')

# 优化坐标与样式
plt.xticks(x, labels, rotation=15, ha='right', fontsize=14)
plt.ylabel('Latency (s)', fontsize=18)
plt.title('Page Loading Latency (20%-80% Percentile)', fontsize=18, fontweight='bold', pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
plt.ylim(y_min, y_max)

# 添加图例（自定义patch）
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#76B7B2', edgecolor='#333333', label='20%-80% Percentile'),
    Line2D([0], [0], color='#1B4F72', lw=2.2, label='Median')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=15, frameon=False)

plt.tight_layout()
plt.savefig('latency.pdf', dpi=400, bbox_inches='tight')
plt.show()
