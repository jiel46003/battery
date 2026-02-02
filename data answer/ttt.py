import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. 物理参数与环境初始化 (Initialization)
# ==========================================
print("🚀 初始化模型参数与数据...")

# --- A. 基础物理曲线 (Task 2 成果) ---
# OCV & R 曲线 (Baseline @ 25°C, New Battery)
SOC_BASE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OCV_BASE = [3.05, 3.25, 3.45, 3.60, 3.70, 3.76, 3.82, 3.89, 3.96, 4.06, 4.16, 4.25]
R_BASE = [0.200, 0.150, 0.100, 0.080, 0.060, 0.050, 0.050, 0.050, 0.050, 0.055, 0.060, 0.065]

# --- B. 智能加载用户行为数据 ---
# 尝试寻找上一问生成的真实行为数据，找不到则使用模拟数据兜底
possible_paths = ['model_ready_data (1).csv', 'data/model_ready_data (1).csv', '../model_ready_data (1).csv']
df_user = None
for path in possible_paths:
    if os.path.exists(path):
        try:
            df_user = pd.read_csv(path)
            print(f"✅ 已加载真实用户行为数据: {path}")
            break
        except:
            continue

if df_user is not None:
    # 归一化昼夜因子 (Mean=1)
    DIURNAL_PATTERN = df_user['diurnal_factor'].values / df_user['diurnal_factor'].mean()
else:
    print("⚠️ 未找到数据文件，使用正弦波模拟昼夜节律...")
    t = np.linspace(0, 24, 288)
    DIURNAL_PATTERN = 1.0 + 0.6 * np.sin((t - 9) * np.pi / 12)


# ==========================================
# 2. 扩展物理模型 (Extended Model Class)
# ==========================================
class ExtendedBatteryModel:
    def __init__(self, temp_c=25.0, cycle_count=0, cutoff_v=3.0):
        """
        初始化电池模型，支持环境与寿命参数注入
        注意：此处只接收物理参数，不接收 p_game
        :param temp_c: 温度 (°C) -> 影响内阻
        :param cycle_count: 循环次数 (N) -> 影响容量 & 内阻
        :param cutoff_v: 截止电压 (V) -> 影响终止条件
        """
        self.soc = 1.0
        self.voltage = 4.2
        self.is_dead = False
        self.cutoff_v = cutoff_v

        # --- 物理修正逻辑 (Physics Corrections) ---

        # 1. 容量老化修正 (Capacity Fade)
        # 假设 1000 次循环衰减 20% (基于平方根定律 SEI Growth)
        aging_factor_cap = 1.0 - 0.20 * np.sqrt(cycle_count / 1000.0)
        self.capacity = 15.0 * aging_factor_cap  # Wh

        # 2. 内阻温度与老化修正 (Resistance Correction)
        # Arrhenius 温度项: exp( E_a/R * (1/T - 1/T_ref) )
        tk = temp_c + 273.15
        temp_factor = np.exp(2500 * (1 / tk - 1 / 298.15))
        # 老化内阻项: 线性增加
        aging_factor_res = 1.0 + 0.5 * (cycle_count / 1000.0)

        total_r_scale = temp_factor * aging_factor_res

        # 构建修正后的插值函数
        # 只有内阻 R 随温度变化显著，OCV 变化较小忽略
        r_adjusted = [r * total_r_scale for r in R_BASE]
        self.f_ocv = interp1d(SOC_BASE, OCV_BASE, kind='cubic', fill_value="extrapolate")
        self.f_r = interp1d(SOC_BASE, r_adjusted, kind='cubic', fill_value="extrapolate")

    def step(self, power_w, dt_sec):
        if self.is_dead: return

        # 物理计算
        voc = self.f_ocv(self.soc)
        r_int = self.f_r(self.soc)

        # 迭代求解端电压 V = Voc - (P/V)*R
        v_guess = voc
        for _ in range(3):
            # 避免除以0
            if v_guess < 0.1: v_guess = 0.1
            i_load = power_w / v_guess
            v_guess = voc - i_load * r_int

        self.voltage = v_guess

        # 积分 SOC
        # Power(W) * Time(h) / Capacity(Wh)
        self.soc -= (power_w * dt_sec / 3600.0) / self.capacity

        # 终止判定
        # 1. 电量耗尽 OR 2. 电压过低保护
        if self.soc <= 0.005 or self.voltage <= self.cutoff_v:
            self.is_dead = True


# ==========================================
# 3. 敏感性分析引擎 (Sensitivity Engine)
# ==========================================
def run_sensitivity_test(variable_name, value_range, n_sims=50):
    """
    控制变量法测试引擎
    """
    results_tte = []

    # 默认基准参数 (Baseline)
    # 包含物理参数和行为参数
    base_params = {
        'temp_c': 25.0,
        'cycle_count': 100,
        'cutoff_v': 3.0,
        'p_game': 0.15
    }

    print(f"⚡ 测试变量: {variable_name} | 范围: {value_range[0]:.1f} -> {value_range[-1]:.1f}...")

    for val in value_range:
        # 1. 复制基准参数
        params = base_params.copy()

        # 2. 更新当前测试的变量值
        # 无论 variable_name 是物理参数还是行为参数，直接更新字典
        params[variable_name] = val

        # 3. [关键修复] 分离参数
        # p_game 是行为参数，用于 while 循环，不能传给 Model.__init__
        # pop() 方法会将其从字典中移除并返回其值
        p_game_val = params.pop('p_game')

        # 4. 运行模拟 (多次取平均以消除随机性)
        ttes = []
        for _ in range(n_sims):
            # 初始化物理模型 (此时 params 只剩 temp_c, cycle_count, cutoff_v)
            model = ExtendedBatteryModel(**params)

            t_elapsed = 0
            idx = np.random.randint(0, 288)  # 随机开始时间

            # 模拟直到关机或超时 (3天)
            while not model.is_dead and t_elapsed < 86400 * 3:
                # --- 行为模拟 (MCMC 简化逻辑) ---
                # 考虑昼夜节律
                diurnal = DIURNAL_PATTERN[idx % len(DIURNAL_PATTERN)]
                prob_active = np.clip(0.35 * diurnal, 0.01, 0.99)

                # 状态判定
                if np.random.random() > prob_active:
                    # 待机状态
                    p_load = 0.15
                    dt = 1800
                else:
                    # 活跃状态：判定是否打游戏 (High Load)
                    if np.random.random() < p_game_val:
                        p_load = 6.5;
                        dt = 900  # 游戏 15min
                    else:
                        p_load = 1.2;
                        dt = 60  # 微信 1min

                dt = min(dt, 300)  # 时间步限制

                # --- 物理步进 ---
                model.step(p_load, dt)
                t_elapsed += dt
                idx += 1

            ttes.append(t_elapsed / 3600.0)

        # 记录该变量值下的平均 TTE
        results_tte.append(np.mean(ttes))

    return np.array(results_tte)


# ==========================================
# 4. 执行分析与数据保存 (Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 温度敏感性 (Temperature)
    # 范围: -20°C (极寒) 到 40°C (酷暑)
    temps = np.linspace(-20, 40, 10)
    res_temp = run_sensitivity_test('temp_c', temps)

    # 2. 老化敏感性 (Cycle Count)
    # 范围: 0 (新机) 到 1000 (报废边缘)
    cycles = np.linspace(0, 1000, 10)
    res_cycle = run_sensitivity_test('cycle_count', cycles)

    # 3. 行为敏感性 (Gaming Probability)
    # 范围: 0% (不玩) 到 50% (重度成瘾)
    probs = np.linspace(0, 0.5, 10)
    res_prob = run_sensitivity_test('p_game', probs)

    # 4. 机理敏感性 (Cutoff Voltage)
    # 范围: 2.8V (深放电) 到 3.4V (浅放电保护)
    cutoffs = np.linspace(2.8, 3.4, 10)
    res_cutoff = run_sensitivity_test('cutoff_v', cutoffs)

    # 保存数据用于可视化
    np.savez("sensitivity_data.npz",
             temps=temps, res_temp=res_temp,
             cycles=cycles, res_cycle=res_cycle,
             probs=probs, res_prob=res_prob,
             cutoffs=cutoffs, res_cutoff=res_cutoff)

    print("\n✅ 所有敏感性测试完成！")
    print("📂 数据已保存至: sensitivity_data.npz")
    print("💡 接下来请运行可视化脚本生成图表。")
