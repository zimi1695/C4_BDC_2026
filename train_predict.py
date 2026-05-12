# ============================================================
# 文件名: train_predict.py
# 放置位置: regression-lgbm/train_predict.py
# 作用: 从stock_data.csv出发，完成回归预测→Top5选股→收益评估
# ============================================================

import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ============================================================
# 【配置区】所有需要你修改的参数都在这里
# ============================================================

# 数据文件路径（相对于本脚本的位置）
DATA_PATH = "./data/stock_data.csv"

# 输出结果保存路径
OUTPUT_DIR = "./output"

# 训练集结束时间 / 测试集开始时间（按时间切割，绝对不能重叠！）
TRAIN_END   = "2024-06-30"
TEST_START  = "2024-07-01"

# 预测目标：未来几个交易日的收益率（5 = 一周）
FORWARD_DAYS = 5

# 每周选几只股票
TOP_K = 5

# ============================================================
# 【第一部分】读取并清洗数据
# ============================================================

def load_data(path):
    print("=" * 50)
    print("【第一步】读取数据...")

    df = pd.read_csv(path, encoding="utf-8")

    # 如果utf-8报错，改成下面这行（GBK编码）:
    # df = pd.read_csv(path, encoding="gbk")

    print(f"  原始数据：{df.shape[0]} 行，{df.shape[1]} 列")
    print(f"  原始列名：{df.columns.tolist()}")

    # ---- 列名标准化（把中文列名映射到英文）----
    df = df.rename(columns={
        "股票代码": "stock_code",
        "日期":   "date",
        "开盘":   "open",
        "收盘":   "close",
        "最高":   "high",
        "最低":   "low",
        "成交量":  "volume",
        "成交额":  "amount",
        "振幅":   "amplitude",
        "涨跌额":  "price_change",
        "换手率":  "turnover",
        "涨跌幅":  "pct_change",
    })

    # ---- 清洗股票代码：去掉前面的单引号 ' ----
    # '000001 → 000001
    df["stock_code"] = (
        df["stock_code"]
        .astype(str)
        .str.replace("'", "")
        .str.strip()
        .str.zfill(6)
    )
    print(f"  股票代码示例（清洗后）：{df['stock_code'].unique()[:5]}")

    # ---- 日期格式统一 ----
    # 2020/1/10 → 2020-01-10（pandas标准格式）
    df["date"] = pd.to_datetime(df["date"])

    # ---- 数据类型转换（确保价格/成交量是数字）----
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 排序 ----
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # ---- 基本统计 ----
    print(f"  清洗后：{df.shape[0]} 行")
    print(f"  股票数量：{df['stock_code'].nunique()} 只")
    print(f"  时间范围：{df['date'].min()} ~ {df['date'].max()}")

    return df


# ============================================================
# 【第二部分】特征工程：计算技术指标
# ============================================================

def compute_features(df):
    print("=" * 50)
    print("【第二步】计算技术指标特征...")

    result_dfs = []

    # 对每只股票单独计算（因为不同股票不能混在一起算均线）
    grouped = df.groupby("stock_code")
    total = len(grouped)

    for i, (stock_code, g) in enumerate(grouped):
        g = g.copy().sort_values("date")

        close  = g["close"]
        high   = g["high"]
        low    = g["low"]
        volume = g["volume"]

        # ---- 均线特征 ----
        g["ma5"]  = close.rolling(5).mean()
        g["ma10"] = close.rolling(10).mean()
        g["ma20"] = close.rolling(20).mean()

        # ---- 价格相对均线比值（去量纲，不同股票价格可以比较）----
        g["price_ma5_ratio"]  = close / (g["ma5"]  + 1e-8)
        g["price_ma10_ratio"] = close / (g["ma10"] + 1e-8)
        g["price_ma20_ratio"] = close / (g["ma20"] + 1e-8)

        # ---- 历史收益率特征 ----
        g["ret_1d"]  = close.pct_change(1)
        g["ret_5d"]  = close.pct_change(5)
        g["ret_10d"] = close.pct_change(10)
        g["ret_20d"] = close.pct_change(20)

        # ---- 波动率（收益率的标准差）----
        g["vol_5d"]  = g["ret_1d"].rolling(5).std()
        g["vol_10d"] = g["ret_1d"].rolling(10).std()
        g["vol_20d"] = g["ret_1d"].rolling(20).std()

        # ---- 成交量相对特征 ----
        g["vol_ratio_5d"]  = volume / (volume.rolling(5).mean()  + 1e-8)
        g["vol_ratio_10d"] = volume / (volume.rolling(10).mean() + 1e-8)

        # ---- 价格位置（今天的收盘价在近20天最高最低中的位置）----
        roll_high = high.rolling(20).max()
        roll_low  = low.rolling(20).min()
        g["price_position_20d"] = (close - roll_low) / (roll_high - roll_low + 1e-8)

        # ---- RSI（相对强弱指标，衡量超买超卖）----
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        g["rsi_14"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-8))

        # ---- MACD（趋势跟踪指标）----
        ema12          = close.ewm(span=12, adjust=False).mean()
        ema26          = close.ewm(span=26, adjust=False).mean()
        g["macd"]        = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"]   = g["macd"] - g["macd_signal"]

        # ---- 换手率特征（你的数据里有！直接用）----
        g["turnover_ratio_5d"] = g["turnover"] / (g["turnover"].rolling(5).mean() + 1e-8)

        result_dfs.append(g)

        # 进度提示
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  进度：{i+1}/{total} 只股票")

    result = pd.concat(result_dfs, ignore_index=True)
    print("  特征计算完成！")
    return result


# ============================================================
# 【第三部分】构造标签：未来5个交易日的收益率
# ============================================================

def compute_label(df, forward_days=5):
    print("=" * 50)
    print(f"【第三步】构造标签（未来{forward_days}天收益率）...")

    result_dfs = []

    for stock_code, g in df.groupby("stock_code"):
        g = g.copy().sort_values("date")

        # 核心公式：未来第forward_days天的收盘价 / 今天收盘价 - 1
        # shift(-5) = 把5行之后的值移到当前行
        g["label"] = g["close"].shift(-forward_days) / g["close"] - 1

        result_dfs.append(g)

    result = pd.concat(result_dfs, ignore_index=True)

    # 最后forward_days天没有未来数据，标签是NaN，删掉
    before = len(result)
    result = result.dropna(subset=["label"])
    after  = len(result)
    print(f"  删除无标签行：{before - after} 行（正常现象）")
    print(f"  标签统计：均值={result['label'].mean():.4f}，标准差={result['label'].std():.4f}")

    return result


# ============================================================
# 【第四部分】划分训练集/测试集 + 准备特征矩阵
# ============================================================

# 所有特征列（和第二部分计算的完全对应）
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

def prepare_dataset(df, train_end, test_start):
    print("=" * 50)
    print("【第四步】划分训练集/测试集...")

    # 删除特征中有NaN的行（均线在最初几行计算不出来）
    df_clean = df.dropna(subset=FEATURE_COLS + ["label"]).copy()

    train_df = df_clean[df_clean["date"] <= train_end].copy()
    test_df  = df_clean[df_clean["date"] >= test_start].copy()

    print(f"  训练集：{len(train_df):,} 条 | {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"  测试集：{len(test_df):,}  条 | {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"].values
    X_test  = test_df[FEATURE_COLS]

    return X_train, y_train, X_test, train_df, test_df


# ============================================================
# 【第五部分】训练LightGBM模型
# ============================================================

def train_model(X_train, y_train):
    print("=" * 50)
    print("【第五步】训练LightGBM回归模型...")

    model = LGBMRegressor(
        n_estimators=500,        # 迭代轮数（树的数量）
        learning_rate=0.05,      # 学习率
        num_leaves=64,           # 叶子节点数（控制复杂度）
        min_child_samples=20,    # 叶子最小样本数（防过拟合）
        subsample=0.8,           # 样本采样比例
        colsample_bytree=0.8,    # 特征采样比例
        reg_alpha=0.1,           # L1正则化
        reg_lambda=0.1,          # L2正则化
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)
    print("  模型训练完成！")

    # 打印特征重要性（看看哪些特征对预测最有用）
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\n  Top10 重要特征：")
    print(importance.head(10).to_string())

    return model


# ============================================================
# 【第六部分】预测 + 排序 + 每周选Top5
# ============================================================

def predict_and_rank(model, X_test, test_df):
    print("=" * 50)
    print("【第六步】预测评分并每周选Top5...")

    test_df = test_df.copy()
    test_df["pred_score"] = model.predict(X_test)

    # 按"周"分组（每周第一个交易日作为调仓日）
    test_df["week_period"] = test_df["date"].dt.to_period("W")

    selection_records = []  # 记录每周的选股结果

    for week, week_data in test_df.groupby("week_period"):
        # 取这一周最早的交易日
        rebalance_date = week_data["date"].min()
        day_data = week_data[week_data["date"] == rebalance_date].copy()

        # 按预测分数降序，选Top5
        top5 = day_data.nlargest(TOP_K, "pred_score")[
            ["stock_code", "pred_score", "label"]
        ].copy()
        top5["week"]           = str(week)
        top5["rebalance_date"] = rebalance_date
        top5["weight"]         = 1.0 / TOP_K  # 等权重，每只20%

        selection_records.append(top5)

    selections = pd.concat(selection_records, ignore_index=True)
    print(f"  共选出 {selections['week'].nunique()} 周 × {TOP_K} 只 = {len(selections)} 条记录")

    return selections


# ============================================================
# 【第七部分】评估收益并保存结果
# ============================================================

def evaluate_and_save(selections, output_dir):
    print("=" * 50)
    print("【第七步】评估收益...")

    os.makedirs(output_dir, exist_ok=True)

    # 每周组合收益 = Top5股票实际收益率的等权平均
    weekly_summary = (
        selections
        .groupby(["week", "rebalance_date"])
        .agg(
            portfolio_return=("label", "mean"),       # 组合实际收益
            avg_pred_score  =("pred_score", "mean"),  # 平均预测分数
            stocks          =("stock_code", list),    # 选的股票列表
        )
        .reset_index()
        .sort_values("rebalance_date")
    )

    # 打印每周明细
    print("\n  每周选股结果：")
    print(f"  {'周':^20} {'日期':^12} {'实际收益':^10} {'股票组合'}")
    print("  " + "-" * 70)
    for _, row in weekly_summary.iterrows():
        stocks_str = ", ".join(row["stocks"])
        print(f"  {row['week']:^20} {str(row['rebalance_date'].date()):^12} "
              f"{row['portfolio_return']:^10.2%} {stocks_str}")

    # ---- 汇总指标计算 ----
    rets = weekly_summary["portfolio_return"].values
    n    = len(rets)

    cumulative_return  = np.prod(1 + rets) - 1
    annualized_return  = (1 + cumulative_return) ** (52 / max(n, 1)) - 1
    sharpe_ratio       = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(52)
    win_rate           = (rets > 0).mean()

    # 最大回撤
    cum_values = np.cumprod(1 + rets)
    peak       = np.maximum.accumulate(cum_values)
    max_drawdown = ((cum_values - peak) / (peak + 1e-8)).min()

    print("\n" + "=" * 50)
    print("  📊 绩效汇总")
    print("=" * 50)
    print(f"  总周数       : {n} 周")
    print(f"  累计收益率   : {cumulative_return:+.2%}")
    print(f"  年化收益率   : {annualized_return:+.2%}")
    print(f"  周均收益率   : {np.mean(rets):+.4%}")
    print(f"  夏普比率     : {sharpe_ratio:.3f}")
    print(f"  最大回撤     : {max_drawdown:.2%}")
    print(f"  周胜率       : {win_rate:.2%}")
    print("=" * 50)

    # ---- 保存结果到CSV ----
    out_path = os.path.join(output_dir, "top5_selections.csv")
    selections.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ 选股明细已保存至：{out_path}")

    summary_path = os.path.join(output_dir, "weekly_summary.csv")
    weekly_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ 周度汇总已保存至：{summary_path}")

    return weekly_summary




def generate_submission(model, df, output_dir):
    print("=" * 50)
    print("【第八步】生成比赛提交文件...")
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    latest_date = df_clean["date"].max()
    print(f"  数据最新日期：{latest_date.date()}")
    latest_data = df_clean[df_clean["date"] == latest_date].copy()
    print(f"  可用股票数：{len(latest_data)} 只")
    X_latest = latest_data[FEATURE_COLS]
    latest_data = latest_data.copy()
    latest_data["pred_score"] = model.predict(X_latest)
    top5 = latest_data.nlargest(TOP_K, "pred_score")[
        ["stock_code", "pred_score"]
    ].copy()
    top5["rank"]   = range(1, TOP_K + 1)
    top5["weight"] = 1.0 / TOP_K
    print("\n  ===== 下周推荐的Top5股票 =====")
    print(f"  {'排名':<6} {'股票代码':<12} {'预测分数':<12} {'建议权重'}")
    print("  " + "-" * 45)
    for _, row in top5.iterrows():
        print(f"  {int(row['rank']):<6} {row['stock_code']:<12} "
              f"{row['pred_score']:<12.6f} {row['weight']:.0%}")
    os.makedirs(output_dir, exist_ok=True)
    submit_path = os.path.join(output_dir, "submission.csv")
    top5[["rank", "stock_code", "weight", "pred_score"]].to_csv(
        submit_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n  ✅ 提交文件已保存：{submit_path}")
    return top5




# ============================================================
# 【主程序】按顺序调用以上所有步骤
# ============================================================

if __name__ == "__main__":

    # Step 1: 读取数据
    df = load_data(DATA_PATH)

    # Step 2: 计算技术指标
    df = compute_features(df)

    # Step 3: 构造标签（预测未来5天收益）
    df = compute_label(df, forward_days=FORWARD_DAYS)

    # Step 4: 划分训练/测试集
    X_train, y_train, X_test, train_df, test_df = prepare_dataset(
        df, TRAIN_END, TEST_START
    )

    # Step 5: 训练模型
    model = train_model(X_train, y_train)

    # Step 6: 预测 + 选Top5
    selections = predict_and_rank(model, X_test, test_df)

    # Step 7: 评估 + 保存
    summary = evaluate_and_save(selections, OUTPUT_DIR)

    submission = generate_submission(model, df, OUTPUT_DIR)
    
    print("\n🎉 全部完成！提交文件在 ./output/submission.csv")
    print("\n🎉 去 ./output 文件夹查看结果吧！")