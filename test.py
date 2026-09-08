import pandas as pd
import os

def extract_test_data(data_path='data/finaldata.csv', output_path='data/test_data.csv'):
    # ==========================================
    # 1. 严格对应原模型中的配置参数
    # ==========================================
    time_interval = '30min'
    history_steps = 12
    future_steps = 1
    train_ratio = 0.7
    val_ratio = 0.15
    # (隐含 test_ratio = 0.15)

    if not os.path.exists(data_path):
        print(f"【错误】找不到原始数据文件: {data_path}")
        return

    print(f"正在读取原始数据: {data_path}...")
    df = pd.read_csv(data_path)
    df['dep_time'] = pd.to_datetime(df['dep_time'])

    # ==========================================
    # 2. 复现原模型的时间聚合逻辑，获取完整时间索引
    # ==========================================
    print("正在计算时间轴并定位测试集断点...")

    # 按照 30min 聚合，仅为了获取首尾时间戳以生成全量时间范围
    time_grouped = df.groupby(pd.Grouper(key='dep_time', freq=time_interval)).size()
    full_idx = pd.date_range(start=time_grouped.index[0], end=time_grouped.index[-1], freq=time_interval)

    # N: 总时间步数量
    N = len(full_idx)
    # M: 滑动窗口截取后的有效样本总数量
    M = N - history_steps - future_steps + 1

    train_end = int(M * train_ratio)
    val_end = train_end + int(M * val_ratio)

    # 测试集第一个样本在全量时间轴 full_idx 中的索引是 val_end
    test_history_start_time = full_idx[val_end]
    # 测试集第一个样本的预测目标的时间点
    test_predict_start_time = full_idx[val_end + history_steps]

    print("-" * 50)
    print(f"全量时间步总数 (N): {N}")
    print(f"有效样本总数 (M): {M}")
    print(f"测试集样本起始索引: {val_end}")
    print(f"-> 测试集(包含前置历史输入) 起始时间: {test_history_start_time}")
    print(f"-> 测试集(纯模型预测目标)   起始时间: {test_predict_start_time}")
    print("-" * 50)

    # ==========================================
    # 3. 过滤并提取测试集对应时间段的原始数据
    # ==========================================
    # 注意：这里默认保留模型推理所需的 12 个历史时间步的数据。
    # 如果你只想要【纯预测区间】的数据，可以将下面的变量改为 test_predict_start_time
    cutoff_time = test_history_start_time

    print(f"\n正在提取 {cutoff_time} 及之后的原始订单数据...")
    test_df = df[df['dep_time'] >= cutoff_time]

    # ==========================================
    # 4. 保存为新的 CSV
    # ==========================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)

    print(f"✅ 提取完成！")
    print(f"测试集数据已保存至: {output_path}")
    print(f"原始数据总行数: {len(df)}")
    print(f"提取出测试集行数: {len(test_df)} (占比: {len(test_df)/len(df):.2%})")

if __name__ == "__main__":
    extract_test_data()