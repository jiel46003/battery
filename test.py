import sys
import os

# --- [关键检查] 防止文件名冲突 ---
if os.path.basename(__file__) == 'matplotlib.py':
    print(" ❌  错误：请将脚本文件名从 'matplotlib.py' 改为其他名字，否则会覆盖标准库！")
    sys.exit(1)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

# ==========================================
# 0. 全自动配置与风格定义
# ==========================================
BASE_DIR = os.getcwd()
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FILE_USER_LOG = os.path.join(DATA_RAW_DIR, 'user_behavior_raw.csv')
FILE_BATTERY_PROFILE = os.path.join(DATA_RAW_DIR, 'battery_ocv_r_profile.csv')
FILE_OUTPUT = os.path.join(DATA_PROCESSED_DIR, 'model_ready_data.csv')
PLOT_OUTPUT = os.path.join(DATA_PROCESSED_DIR, 'data_preprocessing_report_final.png')

MAX_VOLTAGE = 4.45
MIN_VOLTAGE = 2.50

# --- 设置专业绘图风格 (Academic Style) ---
try:
    matplotlib.rcParams.update({
        "font.family": ['Times New Roman', 'Arial', 'sans-serif'],
        "mathtext.fontset": 'stix',
        "axes.unicode_minus": False,
        "xtick.direction": 'in',
        "ytick.direction": 'in',  # 确保这里有逗号
    })
except Exception as e:
    print(f" ⚠️  字体配置警告: {e}")

# 设置 Seaborn 主题
sns.set_theme(style="ticks", font_scale=1.1, rc={
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": ".85",
    "grid.linestyle": "--",  # 确保字典末尾符号闭合正确
})

ACADEMIC_COLORS = ["#004c99", "#c0392b", "#27ae60", "#8e44ad"]
sns.set_palette(sns.color_palette(ACADEMIC_COLORS))


# ==========================================
# 1. 自动化环境与数据初始化
# ==========================================
def initialize_environment():
    for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f" 📁  文件夹已检查: {directory}")

    if not os.path.exists(FILE_BATTERY_PROFILE):
        print(" ⚡  生成标准电池特性表 (Source: Wang et al., 2015)...")
        ocv_data = {
            'soc_percent': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'ocv_voltage': [3.20, 3.35, 3.50, 3.65, 3.72, 3.78, 3.82, 3.88, 3.95, 4.05, 4.15, 4.20],
            'internal_resistance': [0.150, 0.120, 0.080, 0.060, 0.055, 0.050, 0.045, 0.045, 0.045, 0.048, 0.055, 0.060]
        }
        pd.DataFrame(ocv_data).to_csv(FILE_BATTERY_PROFILE, index=False)

    if not os.path.exists(FILE_USER_LOG):
        print("\n ⚠️  警告: 未检测到 'user_behavior_raw.csv'。")
        raise FileNotFoundError("Missing raw data file: user_behavior_raw.csv")


# ==========================================
# 2. 预处理核心逻辑
# ==========================================
def preprocess_battery_data(df, ocv_df):
    df_clean = df.copy()
    print("\n[预处理执行中...]")

    # 2.1 缺失值插值
    if df_clean['voltage_v'].isnull().sum() > 0:
        original_missing = df_clean['voltage_v'].isnull().sum()
        df_clean['voltage_v'] = df_clean['voltage_v'].interpolate(method='linear')
        print(f"   -> 修复电压信号缺失: {original_missing} 处")

    # 2.2 物理界限过滤
    mask_valid_v = (df_clean['voltage_v'] <= MAX_VOLTAGE) & (df_clean['voltage_v'] >= MIN_VOLTAGE)
    dropped_count = len(df_clean) - mask_valid_v.sum()
    df_clean = df_clean[mask_valid_v]
    if dropped_count > 0:
        print(f"   -> 剔除物理异常点: {dropped_count} 个 (电压越界)")

    # 2.3 特征工程
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean['hour'] = df_clean['timestamp'].dt.hour

    # 昼夜因子计算
    df_clean['diurnal_factor'] = 0.3 + 0.6 * np.sin((df_clean['hour'] - 6) * np.pi / 24) ** 2
    df_clean['diurnal_factor'] = df_clean['diurnal_factor'] * (1 + 0.3 * np.sin((df_clean['hour'] - 18) * np.pi / 6))
    df_clean['diurnal_factor'] = df_clean['diurnal_factor'].clip(0.1, 1.5)

    # 2.4 三次样条插值
    f_ocv = interp1d(ocv_df['soc_percent'], ocv_df['ocv_voltage'], kind='cubic', fill_value="extrapolate")
    f_r = interp1d(ocv_df['soc_percent'], ocv_df['internal_resistance'], kind='cubic', fill_value="extrapolate")

    return df_clean, f_ocv, f_r


# ==========================================
# 3. 专业级可视化分析
# ==========================================
def plot_data_quality_pro(raw_df, clean_df, ocv_df):
    # 创建 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.suptitle('Data Preprocessing & Physical Characteristic Analysis Report',
                 fontsize=20, fontweight='bold', y=0.96)

    # --- Fig 1: 电压信号清洗对比 ---
    ax1 = axes[0, 0]

    # 【修复重点】确保多行参数之间有逗号
    ax1.plot(
        raw_df.index[:300],
        raw_df['voltage_v'][:300],
        color='red',  # 修改为红色
        alpha=0.5,  # 增加不透明度
        linestyle=':',
        linewidth=1.5,  # 增加线宽
        label='Raw Signal (Noisy)'
    )

    # 绘制清洗后的数据
    ax1.plot(clean_df.index[:300], clean_df['voltage_v'][:300],
             color=ACADEMIC_COLORS[0], linewidth=2, label='Cleaned Signal')

    ax1.set_title('Fig 1. Voltage Signal Restoration', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Terminal Voltage (V)')
    ax1.set_xlabel('Sample Index (First 300 Points)')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')

    # --- Fig 2: 昼夜活跃因子提取 ---
    ax2 = axes[0, 1]
    hourly_avg = clean_df.groupby('hour')['diurnal_factor'].mean()
    ax2.fill_between(hourly_avg.index, hourly_avg.values, color=ACADEMIC_COLORS[2], alpha=0.2)
    ax2.plot(hourly_avg.index, hourly_avg.values, color=ACADEMIC_COLORS[2], marker='o', linewidth=2, markersize=6)
    ax2.set_title('Fig 2. Extracted Diurnal Activity Factor D(t)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Normalized Activity Factor')
    ax2.set_xlabel('Hour of Day')
    ax2.set_xticks(range(0, 25, 4))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax2.set_ylim(0, 1.6)

    # --- Fig 3: 电池内阻物理特性 ---
    ax3 = axes[1, 0]
    soc_percent = ocv_df['soc_percent'] * 100
    r_values = ocv_df['internal_resistance'] * 1000  # 转换为 mΩ
    ax3.plot(soc_percent, r_values, color=ACADEMIC_COLORS[1], linewidth=2.5, marker='s', markersize=5)
    ax3.axvspan(0, 15, color=ACADEMIC_COLORS[1], alpha=0.15)
    ax3.text(20, r_values.max() * 0.8, "Critical Region:\nHigh IR Drop",
             ha='left', color=ACADEMIC_COLORS[1], fontweight='bold', fontsize=12)
    ax3.set_title('Fig 3. Internal Resistance Profile R(SOC)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Internal Resistance (mΩ)')
    ax3.set_xlabel('State of Charge (%)')
    ax3.invert_xaxis()
    ax3.grid(True, which='both', linestyle='--')

    # --- Fig 4: App 使用频率 ---
    ax4 = axes[1, 1]
    app_counts = clean_df['app_name'].value_counts(normalize=True) * 100
    top_apps = app_counts.head(8)
    sns.barplot(x=top_apps.values, y=top_apps.index, ax=ax4, palette='viridis')
    for i, v in enumerate(top_apps.values):
        ax4.text(v + 0.5, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
    ax4.set_title('Fig 4. App Usage Frequency (Pareto Principle)', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Frequency of Appearance (%)')
    ax4.set_ylabel('')
    ax4.set_xlim(0, top_apps.max() * 1.2)
    ax4.grid(axis='x')

    # 保存
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f" 📊  专业级图表已保存至: {PLOT_OUTPUT}")


# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    try:
        initialize_environment()
        print(" 📚  正在加载数据...")
        df_raw = pd.read_csv(FILE_USER_LOG)
        df_ocv = pd.read_csv(FILE_BATTERY_PROFILE)

        df_ready, func_ocv, func_r = preprocess_battery_data(df_raw, df_ocv)

        cols_to_save = ['timestamp', 'app_name', 'duration_s', 'voltage_v', 'diurnal_factor']
        df_ready[cols_to_save].to_csv(FILE_OUTPUT, index=False)

        print("\n" + "=" * 40)
        print(" ✅  数据预处理成功 (SUCCESS)")
        print("=" * 40)
        print(f"输出文件: {FILE_OUTPUT}")
        print(f"处理条目: {len(df_ready)}")

        plot_data_quality_pro(df_raw, df_ready, df_ocv)
        print("\n 🎉  处理结束，请查看 data/processed 文件夹。")

    except FileNotFoundError as e:
        print(f"\n ❌  错误: {e}")
    except Exception as e:
        import traceback

        print(f"\n ❌  发生未预期的错误:\n")
        traceback.print_exc()
