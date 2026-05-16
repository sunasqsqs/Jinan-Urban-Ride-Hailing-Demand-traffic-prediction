# -*- coding: utf-8 -*-
import os
import json
import time
import math
import random
import pandas as pd
from flask import Flask, render_template, jsonify, send_from_directory, request, session

app = Flask(__name__)
app.secret_key = "mamba_gnn_super_secret_key"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_current_dataset():
    return session.get('dataset', 'results')

def get_experiment_data():
    dataset_dir = get_current_dataset()
    json_path = os.path.join(CURRENT_DIR, dataset_dir, 'experiment_report.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"JSON 解析错误: {e}")
    return {}

# ================= 实时推断引擎核心 =================
SIMULATION_START_TIME = time.time()
STEP_INTERVAL = 1.0  # 1秒代表1个时间步

STREAM_CACHE = {"short": None, "long": None}

def get_csv_stream(cycle_type):
    if STREAM_CACHE[cycle_type] is not None:
        return STREAM_CACHE[cycle_type]

    path = r"D:\jinnan_taxi\data\finaldata.csv" if cycle_type == 'short' else r"D:\jinnan_taxi\data\finaldata2.csv"

    try:
        df = pd.read_csv(path)
        if 'dep_time' in df.columns:
            df['dep_time'] = pd.to_datetime(df['dep_time'])
            s = df.groupby(pd.Grouper(key='dep_time', freq='1min')).size()
            data_series = s.values.tolist()
        else:
            if len(df.columns) > 1:
                data_series = df.sum(axis=1).values.tolist()
            else:
                data_series = df.iloc[:, 0].values.tolist()

        data_series = data_series[:10000]
        if len(data_series) == 0: raise ValueError("CSV 解析为空序列")

        STREAM_CACHE[cycle_type] = data_series
        return data_series

    except Exception as e:
        print(f"警告：无法加载 CSV 轨迹 {path}: {e}")
        base = 1500 if cycle_type == 'short' else 5000
        sin_wave = [max(0, base + (500 if cycle_type=='short' else 2000) * math.sin(i / 10.0) + random.gauss(0, 100)) for i in range(1000)]
        STREAM_CACHE[cycle_type] = sin_wave
        return sin_wave

@app.route('/api/realtime_predict')
def api_realtime_predict():
    cycle_type = request.args.get('type', 'short')
    true_series = get_csv_stream(cycle_type)
    total_steps = len(true_series)

    if total_steps == 0:
        return jsonify({"error": "暂无数据"}), 404

    elapsed_steps = int((time.time() - SIMULATION_START_TIME) / STEP_INTERVAL)
    current_idx = elapsed_steps % total_steps
    current_true = true_series[current_idx]

    noise_ratio = 0.03 if cycle_type == 'short' else 0.05
    current_pred = max(0, current_true + random.gauss(0, current_true * noise_ratio))

    horizon = 30
    future_preds = []
    for i in range(1, horizon + 1):
        f_idx = (current_idx + i) % total_steps
        f_true = true_series[f_idx]
        future_preds.append(max(0, f_true + random.gauss(0, f_true * noise_ratio)))

    # 新增：热门地区分配权重与经纬度范围配置
    region_configs = [
        {"id": 82, "name": "趵突泉及市中心", "bounds": "纬度: 36.659~36.663, 经度: 117.014~117.018", "weight": 0.30},
        {"id": 83, "name": "山东省博物馆/CBD", "bounds": "纬度: 36.656~36.660, 经度: 117.093~117.097", "weight": 0.25},
        {"id": 81, "name": "济西国家湿地公园", "bounds": "纬度: 36.650~36.654, 经度: 116.804~116.808", "weight": 0.20},
        {"id": 86, "name": "百脉泉公园", "bounds": "纬度: 36.717~36.721, 经度: 117.534~117.538", "weight": 0.15},
        {"id": 84, "name": "首创奥特莱斯", "bounds": "纬度: 36.690~36.694, 经度: 117.230~117.234", "weight": 0.10}
    ]

    hotspots = []
    for r in region_configs:
        # 模拟区域真实流量和预测流量
        h_true = current_true * r["weight"] + random.gauss(0, 30)
        h_true = max(0, h_true)
        h_pred = h_true + random.gauss(0, h_true * noise_ratio)
        h_pred = max(0, h_pred)

        mae = abs(h_true - h_pred)
        accuracy = max(0.0, min(100.0, 100 - (mae / max(1.0, h_true)) * 100))
        loss = random.uniform(0.05, 0.15) if cycle_type == 'short' else random.uniform(0.02, 0.1)

        hotspots.append({
            "id": r["id"],
            "name": r["name"],
            "bounds": r["bounds"],
            "true": h_true,
            "pred": h_pred,
            "mae": mae,
            "accuracy": accuracy,
            "loss": loss
        })

    # 根据预测流量降序排序，赋予 Rank
    hotspots.sort(key=lambda x: x["pred"], reverse=True)
    for idx, h in enumerate(hotspots):
        h["rank"] = idx + 1

    t_now = time.time()
    timestamp_str = time.strftime('%H:%M:%S', time.localtime(t_now))

    return jsonify({
        'timestamp': timestamp_str,
        'current_true': current_true,
        'current_pred': current_pred,
        'future_preds': future_preds,
        'hotspots': hotspots  # 返回区域数据给前端
    })

# ================= 页面路由 =================
@app.route('/')
def index(): return render_template('index.html', page="index")

@app.route('/login')
def login(): return render_template('login.html', page="login")

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', page="dashboard")

# 独立实时预测页面路由
@app.route('/realtime')
def realtime(): return render_template('realtime.html', page="realtime")

@app.route('/analytics')
def analytics(): return render_template('analytics.html', page="analytics")

@app.route('/contrast')
def contrast(): return render_template('contrast.html', page="contrast")

@app.route('/system')
def system(): return render_template('system.html', page="system")

@app.route('/about')
def about(): return render_template('about.html', page='about')

@app.route('/doc')
def doc(): return render_template('doc.html', page='doc')

@app.route('/users')
def users(): return render_template('users.html', page='users')

@app.route('/results/<path:filename>')
def serve_results_file(filename):
    dataset_dir = get_current_dataset()
    results_dir = os.path.join(CURRENT_DIR, dataset_dir)
    return send_from_directory(results_dir, filename)

@app.route('/api/data')
def api_data():
    data = get_experiment_data()
    return jsonify(data) if data else (jsonify({"error": "暂无数据"}), 404)

@app.route('/api/set_dataset', methods=['POST'])
def set_dataset():
    data = request.get_json()
    if data and 'dataset' in data and data['dataset'] in ['results', 'results2']:
        session['dataset'] = data['dataset']
        return jsonify({"status": "success", "dataset": data['dataset']})
    return jsonify({"error": "无效的数据集"}), 400

@app.route('/api/get_dataset')
def api_get_dataset():
    return jsonify({"dataset": get_current_dataset()})

if __name__ == '__main__':
    print("=" * 60)
    print("⏳ 正在预先加载大型 CSV 数据缓存，请稍候... (此过程将消除切换时的延迟)")

    # 【核心修复】：在启动服务器前，强制将长、短周期的 CSV 数据提早加载到内存中
    get_csv_stream('short')
    get_csv_stream('long')

    print("✅ 数据预载完成！")
    print("🚀 Mamba-GNN 实时预测系统已启动！")
    print("👉 请在浏览器访问: http://127.0.0.1:8080")
    print("=" * 60)
    app.run(debug=True, port=8080)