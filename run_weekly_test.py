# ============================================================
# 文件名: run_weekly_test.py
# 作用: 对5个测试周分别训练预测，生成result.csv并打分
# 使用方式: python run_weekly_test.py
# ============================================================

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ============================================================
# 【配置区】
# ============================================================

DATA_PATH  = "./data/stock_data.csv"
OUTPUT_DIR = "./output"
TEMP_DIR   = "./temp"
TOP_K      = 5

# 5个测试周（每周的开始日期和结束日期）
# 格式：(周名称, 预测输入截止日, 测试周开始日, 测试周结束日)
# "预测输入截止日" = 测试周开始前最后一个可用的交易日
TEST_WEEKS = [
    ("第1周 (3.09-3.13)", "2026-03-06", "2026-03-09", "2026-03-13"),
    ("第2周 (3.16-3.20)", "2026-03-13", "2026-03-16", "2026-03-20"),
    ("第3周 (3.23-3.27)", "2026-03-20", "2026-03-23", "2026-03-27"),
    ("第4周 (3.30-4.03)", "2026-03-27", "2026-03-30", "2026-04-03"),
    ("第5周 (4.13-4.17)", "2026-04-11", "2026-04-13", "2026-04-17"),
]

# 特征列（和train_predict.py保持一致）
FEATURE_COLS = [
    "ma5", "ma10", "ma20",
    "price_ma5_ratio", "price_ma10_ratio", "price_ma20_ratio",
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "vol_5d", "vol_10d", "vol_20d",
    "vol_ratio_5d", "vol_ratio_10d",
    "price_position_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "turnover_ratio_5d",
]

# ============================================================
# 【数据读取】只读一次，后面复用
# ============================================================

def load_data(path):
    print("读取原始数据...")
    df = pd.read_csv(path, encoding="utf-8")

    # 如果报错换成：df = pd.read_csv(path, encoding="gbk")

    df = df.rename(columns={
        "股票代码": "stock_code",
        "日期":    "date",
        "开盘":    "open",
        "收盘":    "close",
        "最高":    "high",
        "最低":    "low",
        "成交量":   "volume",
        "成交额":   "amount",
        "振幅":    "amplitude",
        "涨跌额":   "price_change",
        "换手率":   "turnover",
        "涨跌幅":   "pct_change",
    })

    # 修复股票代码（去单引号 + 补前导零）
    df["stock_code"] = (
        df["stock_code"]
        .astype(str)
        .str.replace("'", "")
        .str.strip()
        .str.zfill(6)
    )

    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    print(f"  共 {df['stock_code'].nunique()} 只股票，"
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ============================================================
# 【特征工程】
# ============================================================

def compute_features(df):
    result_dfs = []

    for stock_code, g in df.groupby("stock_code"):
        g = g.copy().sort_values("date")
        close  = g["close"]
        high   = g["high"]
        low    = g["low"]
        volume = g["volume"]

        g["ma5"]  = close.rolling(5).mean()
        g["ma10"] = close.rolling(10).mean()
        g["ma20"] = close.rolling(20).mean()

        g["price_ma5_ratio"]  = close / (g["ma5"]  + 1e-8)
        g["price_ma10_ratio"] = close / (g["ma10"] + 1e-8)
        g["price_ma20_ratio"] = close / (g["ma20"] + 1e-8)

        g["ret_1d"]  = close.pct_change(1)
        g["ret_5d"]  = close.pct_change(5)
        g["ret_10d"] = close.pct_change(10)
        g["ret_20d"] = close.pct_change(20)

        g["vol_5d"]  = g["ret_1d"].rolling(5).std()
        g["vol_10d"] = g["ret_1d"].rolling(10).std()
        g["vol_20d"] = g["ret_1d"].rolling(20).std()

        g["vol_ratio_5d"]  = volume / (volume.rolling(5).mean()  + 1e-8)
        g["vol_ratio_10d"] = volume / (volume.rolling(10).mean() + 1e-8)

        roll_high = high.rolling(20).max()
        roll_low  = low.rolling(20).min()
        g["price_position_20d"] = (close - roll_low) / (roll_high - roll_low + 1e-8)

        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        g["rsi_14"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))

        ema12            = close.ewm(span=12, adjust=False).mean()
        ema26            = close.ewm(span=26, adjust=False).mean()
        g["macd"]        = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"]   = g["macd"] - g["macd_signal"]

        g["turnover_ratio_5d"] = g["turnover"] / (g["turnover"].rolling(5).mean() + 1e-8)

        result_dfs.append(g)

    return pd.concat(result_dfs, ignore_index=True)


# ============================================================
# 【新增】截面标准化：每天对所有股票的特征做Z-Score归一化
# ============================================================

def cross_sectional_normalize(df):
    """
    截面标准化：
    对每一天，把所有股票的同一个特征值，
    转换成"这只股票在今天所有股票中排第几"的标准分
    
    这样模型就能真正比较不同股票之间的强弱
    """
    print("  进行截面标准化...")
    df = df.copy()
    
    # 对每个交易日分组
    def normalize_one_day(group):
        for col in FEATURE_COLS:
            if col in group.columns:
                mean = group[col].mean()
                std  = group[col].std()
                # Z-Score标准化：(x - 均值) / 标准差
                group[col] = (group[col] - mean) / (std + 1e-8)
        return group
    
    df = df.groupby("date", group_keys=False).apply(normalize_one_day)
    print("  截面标准化完成！")
    return df


def compute_label(df, forward_days=5):
    result_dfs = []
    for stock_code, g in df.groupby("stock_code"):
        g = g.copy().sort_values("date")
        g["label"] = g["close"].shift(-forward_days) / g["close"] - 1
        result_dfs.append(g)
    result = pd.concat(result_dfs, ignore_index=True)
    return result.dropna(subset=["label"])


# ============================================================
# 【单周预测】给定截止日期，训练模型，预测Top5
# ============================================================

def predict_one_week(df_feat, cutoff_date, week_name):
    """
    改进版：
    1. 只用截止日前2年的数据训练（滑动窗口，避免旧数据干扰）
    2. 加入截面标准化（让不同股票的特征可以横向比较）
    """
    cutoff_dt = pd.Timestamp(cutoff_date)
    
    # ---- 滑动窗口：只取截止日前2年的数据 ----
    window_start = cutoff_dt - pd.DateOffset(years=2)
    
    df_before = df_feat[
        (df_feat["date"] >= window_start) &
        (df_feat["date"] <= cutoff_dt)
    ].copy()
    
    # ---- 截面标准化 ----
    df_before = cross_sectional_normalize(df_before)
    
    # ---- 构造标签 ----
    df_with_label = compute_label(df_before, forward_days=5)
    train_df = df_with_label.dropna(subset=FEATURE_COLS + ["label"]).copy()

    if len(train_df) == 0:
        print(f"  ❌ {week_name}：训练数据为空！")
        return None

    print(f"  训练窗口：{train_df['date'].min().date()} ~ "
          f"{train_df['date'].max().date()}，共 {len(train_df):,} 条")

    # ---- 训练模型 ----
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"].values

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # ---- 预测输入：截止日当天（或最近一天），同样需要截面标准化 ----
    # 取截止日前30天的数据来做标准化（保证有足够股票数量来算均值标准差）
    df_recent = df_feat[
        (df_feat["date"] >= cutoff_dt - pd.DateOffset(days=30)) &
        (df_feat["date"] <= cutoff_dt)
    ].copy()
    df_recent = cross_sectional_normalize(df_recent)

    # 取每只股票在截止日（或之前最近交易日）的特征
    df_clean = df_recent.dropna(subset=FEATURE_COLS).copy()
    df_avail = df_clean[df_clean["date"] <= cutoff_dt]

    latest_per_stock = (
        df_avail
        .sort_values("date")
        .groupby("stock_code")
        .last()
        .reset_index()
    )

    if len(latest_per_stock) == 0:
        print(f"  ❌ {week_name}：预测数据为空！")
        return None

    X_pred = latest_per_stock[FEATURE_COLS]
    latest_per_stock = latest_per_stock.copy()
    latest_per_stock["pred_score"] = model.predict(X_pred)

    # ---- 选Top5 ----
    top5 = latest_per_stock.nlargest(TOP_K, "pred_score")[
        ["stock_code", "pred_score"]
    ].copy()
    top5["权重"] = 1.0 / TOP_K

    print(f"  选出Top5：{top5['stock_code'].tolist()}")
    return top5


# ============================================================
# 【生成test.csv】从原始数据中裁出测试周的数据
# ============================================================

def generate_test_csv(df_raw, week_start, week_end):
    """
    从原始stock_data中裁出测试周的数据，保存为 data/test.csv
    格式需要包含：股票代码, 日期, 开盘, 收盘
    """
    # 筛选测试周日期范围
    mask = (df_raw["date"] >= week_start) & (df_raw["date"] <= week_end)
    test_data = df_raw[mask].copy()

    # 重命名回中文（score_self.py需要中文列名）
    test_data = test_data.rename(columns={
        "stock_code": "股票代码",
        "date":       "日期",
        "open":       "开盘",
        "close":      "收盘",
        "high":       "最高",
        "low":        "最低",
    })

    # 只保留score_self.py需要的列
    test_data = test_data[["股票代码", "日期", "开盘", "收盘"]]

    test_csv_path = "./data/test.csv"
    test_data.to_csv(test_csv_path, index=False, encoding="utf-8-sig")

    stock_count = test_data["股票代码"].nunique()
    day_count   = test_data["日期"].nunique()
    print(f"  test.csv 已生成：{stock_count} 只股票 × {day_count} 个交易日")

    return test_data


# ============================================================
# 【生成result.csv】把预测结果保存为评分脚本需要的格式
# ============================================================

def generate_result_csv(top5):
    """
    输出格式（score_self.py要求）：
    股票代码,权重
    000001,0.2
    ...
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    result = top5[["stock_code", "权重"]].rename(
        columns={"stock_code": "股票代码"}
    )

    result_path = os.path.join(OUTPUT_DIR, "result.csv")
    result.to_csv(result_path, index=False, encoding="utf-8-sig")

    print(f"  result.csv 内容：")
    print(result.to_string(index=False))

    return result_path


# ============================================================
# 【调用score_self.py】自动打分
# ============================================================

def run_scoring():
    """调用score_self.py，返回得分"""
    try:
        proc = subprocess.run(
            [sys.executable, "score_self.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        output = proc.stdout.strip()
        error  = proc.stderr.strip()

        if error:
            print(f"  ⚠️  评分脚本警告：{error[:200]}")

        # 从输出中提取得分
        for line in output.split("\n"):
            if "加权收益率得分" in line or "Final Score" in line:
                print(f"  📊 {line.strip()}")

        # 读取tmp.csv获取数值得分
        tmp_path = "./temp/tmp.csv"
        if os.path.exists(tmp_path):
            tmp = pd.read_csv(tmp_path)
            score = tmp["Final Score"].iloc[0]
            return float(score)
        else:
            print("  ❌ 未找到temp/tmp.csv，评分失败")
            return None

    except Exception as e:
        print(f"  ❌ 调用评分脚本失败：{e}")
        return None


# ============================================================
# 【主程序】循环跑5个测试周
# ============================================================

if __name__ == "__main__":

    # 1. 读取原始数据
    print("=" * 60)
    df_raw = load_data(DATA_PATH)

    # 2. 计算特征（对所有数据一次性计算，后面按日期切片）
    print("\n计算全量特征（需要一点时间）...")
    df_feat = compute_features(df_raw)
    print("特征计算完成！\n")

    # 3. 汇总结果
    all_scores = []

    # 4. 逐周预测+打分
    for week_name, cutoff_date, week_start, week_end in TEST_WEEKS:
        print("=" * 60)
        print(f"🗓️  {week_name}")
        print(f"   训练数据截止：{cutoff_date}")
        print(f"   测试周范围：{week_start} ~ {week_end}")
        print("-" * 60)

        # Step A: 训练模型并预测Top5
        top5 = predict_one_week(df_feat, cutoff_date, week_name)

        if top5 is None:
            all_scores.append((week_name, None))
            continue

        # Step B: 生成 data/test.csv（测试周实际数据）
        generate_test_csv(df_raw, week_start, week_end)

        # Step C: 生成 output/result.csv（预测结果）
        generate_result_csv(top5)

        # Step D: 调用 score_self.py 打分
        score = run_scoring()
        all_scores.append((week_name, score))

        print(f"  ✅ 本周得分：{score:.4f}" if score is not None else "  ❌ 打分失败")

    # 5. 汇总所有周的结果
    print("\n" + "=" * 60)
    print("📊 五周汇总结果")
    print("=" * 60)
    print(f"  {'测试周':<25} {'加权收益率得分':>15}")
    print("  " + "-" * 42)

    valid_scores = []
    for week_name, score in all_scores:
        if score is not None and score != -999:
            score_str = f"{score:>+.4f} ({score*100:+.2f}%)"
            valid_scores.append(score)
        else:
            score_str = "失败"
        print(f"  {week_name:<25} {score_str:>15}")

    if valid_scores:
        avg = np.mean(valid_scores)
        total = np.sum(valid_scores)
        print("  " + "-" * 42)
        print(f"  {'平均得分':<25} {avg:>+.4f} ({avg*100:+.2f}%)")
        print(f"  {'五周累积得分':<25} {total:>+.4f} ({total*100:+.2f}%)")

    print("=" * 60)
    print("\n✅ 全部完成！")