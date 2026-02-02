import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (基于文献统计规律)
# ==========================================
# 路径配置
BASE_DIR = os.getcwd()
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_FILE = os.path.join(DATA_RAW_DIR, 'user_behavior_raw.csv')

# 模拟时长
DAYS = 7
FREQ = '5min'  # 参考 Wagner et al.  的轮询间隔

# APP 行为画像 (基于 Falaki 的应用流行度指数衰减)
# 格式: (App名, 基础功率W, 流行度权重, 平均时长s, 时长波动sigma)
APP_PROFILES = [
    ('System_Idle', 0.2, 0.0, 3600, 1.0),  # 待机 (特殊处理)
    ('WeChat', 1.5, 0.5, 60, 0.8),  # 高频，短时 (LogNormal mu=ln(60))
    ('TikTok', 2.8, 0.3, 600, 1.2),  # 中频，中长时
    ('HonorOfKings', 5.5, 0.1, 1800, 0.5),  # 低频，超长时 (高耗电，电压杀手)
    ('Camera', 4.0, 0.1, 120, 0.5),  # 偶尔使用
]

# 物理参数
BATTERY_CAPACITY_WH = 15.0  # 约 4000mAh * 3.7V
R_INTERNAL_BASE = 0.05  # 基础内阻 (欧姆)


# ==========================================
# 2. 核心生成逻辑
# ==========================================

def get_diurnal_factor(hour):
    """
    生成昼夜节律因子 (0.1 ~ 1.0)
    模拟人类作息：深夜(0-5点)极低，白天(9-22点)活跃
    """
    # 简单的双正弦合成，模拟早晚高峰
    if 0 <= hour < 6:
        return 0.05  # 深夜睡眠
    else:
        # 白天活跃度波动
        return 0.5 + 0.4 * np.sin((hour - 8) * np.pi / 12)


def generate_session_duration(mean_s, sigma):
    """
    生成符合 Falaki 论文 "长尾分布" 的会话时长
    使用对数正态分布 (Log-Normal)
    """
    mu = np.log(mean_s)
    duration = np.random.lognormal(mu, sigma)
    return max(10, duration)  # 至少10秒


def generate_realistic_data():
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)

    print(f"🚀 开始生成基于 Falaki & Wagner 论文的仿真数据...")

    # 1. 生成时间轴
    dates = pd.date_range(start='2024-02-01', periods=DAYS * 24 * 12, freq=FREQ)
    n_steps = len(dates)

    # 初始化状态列表
    app_list = []
    duration_list = []
    voltage_list = []
    soc_list = []

    # 初始电池状态
    current_soc = 1.0  # 100%
    current_state = 'System_Idle'
    state_remaining_time = 0  # 当前状态还剩多少秒

    # 遍历时间步 (Time-Step Simulation)
    for i, t in enumerate(dates):
        hour = t.hour

        # --- A. 状态机切换逻辑 ---
        if state_remaining_time <= 0:
            # 决定下一个状态
            diurnal_prob = get_diurnal_factor(hour)

            # 判定是 "活跃" 还是 "待机"
            # 依据 Falaki: 活跃概率随昼夜变化
            is_active = np.random.random() < diurnal_prob

            if not is_active:
                # 进入待机 (Off time)
                # 依据 Falaki: Off time 服从 Weibull (这里简化为长 Exponential)
                current_state = 'System_Idle'
                state_remaining_time = np.random.exponential(3600)  # 平均待机1小时
            else:
                # 进入活跃 (On time) - 选择 APP
                # 依据 Falaki: App 流行度服从指数衰减
                apps = [x for x in APP_PROFILES if x[0] != 'System_Idle']
                weights = [x[2] for x in apps]
                weights = np.array(weights) / sum(weights)  # 归一化

                chosen_idx = np.random.choice(len(apps), p=weights)
                chosen_app = apps[chosen_idx]

                current_state = chosen_app[0]
                # 生成符合长尾分布的时长
                state_remaining_time = generate_session_duration(chosen_app[3], chosen_app[4])

        # 记录当前步的数据
        # 注意: 如果 state_remaining_time > 300s (5min), 则当前 5min 都是这个状态
        # 如果 < 300s, 这里做简化处理，假设主要状态为 current_state
        step_duration = 300  # 5min step
        active_duration = min(state_remaining_time, step_duration)
        state_remaining_time -= step_duration

        # --- B. 物理电量模拟 (Physics Simulation) ---
        # 查找当前 App 的功率
        profile = next(x for x in APP_PROFILES if x[0] == current_state)
        power_w = profile[1]

        # 1. SOC 更新 (积分法)
        # Energy (Wh) = Power (W) * Time (h)
        energy_consumed = power_w * (step_duration / 3600.0)
        soc_drop = energy_consumed / BATTERY_CAPACITY_WH
        current_soc -= soc_drop

        # 模拟充电行为 (当电量过低或深夜时)
        if current_soc < 0.15 or (current_soc < 0.8 and 1 <= hour < 5):
            current_soc += 0.05  # 快速充电

        current_soc = np.clip(current_soc, 0.05, 1.0)  # 限制范围

        # 2. 电压计算 (含 OCV 和 IR Drop)
        # V_term = V_ocv(SOC) - I * R
        # 简化 OCV 曲线
        v_ocv = 3.2 + 0.8 * current_soc + 0.1 * (current_soc ** 2)

        # 计算负载电流 I = P / V (近似)
        current_amps = power_w / v_ocv

        # 计算内阻 (低电量时内阻增加 - Wang et al.)
        r_internal = R_INTERNAL_BASE * (1 + 2 * np.exp(-10 * current_soc))

        # 计算端电压
        v_term = v_ocv - (current_amps * r_internal)
        # 添加测量噪声 (Wagner 提到的 unreliable data)
        v_term += np.random.normal(0, 0.01)

        # --- 存入列表 ---
        app_list.append(current_state)
        # 记录 duration 为 "当前 5min 内活跃了多久" (近似)
        # 为了符合预处理代码逻辑，这里填实际活跃时长，如果是 idle 则较长
        duration_list.append(float(active_duration) if current_state != 'System_Idle' else 300.0)
        voltage_list.append(round(v_term, 4))
        soc_list.append(round(current_soc, 4))

    # 3. 构造 DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'app_name': app_list,
        'duration_s': duration_list,  # 这一列现在代表 "在该时间窗内的活跃时长"
        'voltage_v': voltage_list,
        # 'soc_true': soc_list # 这一列可以保留用于验证，但预处理代码不直接读它
    })

    # 4. 制造一些真实的 "脏数据" (参考 Wagner et al.)
    # "Jumps in system-reported uptime" [cite: 2205] -> 导致偶尔电压读数丢失或异常
    print("⚡ 注入传感器噪声与异常值 (模拟真实采集环境)...")

    # 随机设置缺失值 (模拟数据上传失败)
    missing_indices = np.random.choice(df.index, size=int(n_steps * 0.02), replace=False)
    df.loc[missing_indices, 'voltage_v'] = np.nan

    # 随机设置异常值 (模拟传感器故障)
    outlier_indices = np.random.choice(df.index, size=int(n_steps * 0.005), replace=False)
    df.loc[outlier_indices, 'voltage_v'] = 0.0  # 瞬间掉电读数

    # 5. 保存
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 数据已生成: {OUTPUT_FILE}")
    print(f"   - 包含列: {list(df.columns)}")
    print(f"   - 总行数: {len(df)}")

    # 简单绘图验证
    plt.figure(figsize=(10, 4))
    plt.plot(df['voltage_v'].iloc[:500], label='Voltage (V)')
    plt.title('Simulated Voltage Trace (First 500 points)')
    plt.ylabel('Voltage')
    plt.legend()
    # plt.show()


if __name__ == '__main__':
    generate_realistic_data()
