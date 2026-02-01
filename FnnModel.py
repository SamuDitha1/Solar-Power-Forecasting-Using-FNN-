#Imports and Global Config
import os
import warnings
import random
import math
import argparse
import io
from pathlib import Path
import numpy as np
import pandas as pd

#Suppress warnings and set environment variables
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#ML/DL imports
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, Model
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler

#Try GUI (Tkinter) + Matplotlib embedding (TkAgg)
TK_OK = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    TK_OK = True
except Exception:
    TK_OK = False

#Constants and Defaults
TEST_FRAC  = 0.10
VAL_FRAC   = 0.10
MAX_LAG    = 72               # 3-day lookback
ROLLS      = (3, 6, 12, 24, 48)
BATCH      = 32
EPOCHS     = 150
PATIENCE   = 25
LR         = 1e-4
SEED       = 42
DATETIME_CANDIDATES = ["Date"]
NICE_TARGET_NAME = "Power Output"  # used for UI labels
ZERO_TOL   = 1e-6          # treat ≤ this as zero (night)
ERR_PCT_MAX = 20.0         # show % error < 20
HOUR_START = 7             # 07:00 inclusive
HOUR_END   = 15            # 15:00 inclusive


#Utility Functions

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

set_seed()

def find_datetime_col(df: pd.DataFrame):
    low = [c.lower() for c in df.columns]
    for k in DATETIME_CANDIDATES:
        if k.lower() in low:
            return df.columns[low.index(k.lower())]
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[dt_col])
    #Basic time features
    df["hour"]  = dt.dt.hour
    df["dow"]   = dt.dt.dayofweek
    df["doy"]   = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["week"]  = dt.dt.isocalendar().week

    #Cyclical encodings
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7)
    df["doy_sin"]   = np.sin(2*np.pi*df["doy"]/365.25)
    df["doy_cos"]   = np.cos(2*np.pi*df["doy"]/365.25)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    #Solar position approximation
    df["solar_hour"] = np.where(
        (df["hour"] >= 6) & (df["hour"] <= 18),
        np.sin(np.pi * (df["hour"] - 6) / 12),
        0
    )

    #Heuristics & flags
    df["is_peak_solar"]   = ((df["hour"] >= 10) & (df["hour"] <= 14)).astype(int) # .astype(int) is used to convert to integer
    df["is_morning_ramp"] = ((df["hour"] >= 7)  & (df["hour"] <= 9)).astype(int)
    df["is_evening_ramp"] = ((df["hour"] >= 15) & (df["hour"] <= 17)).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_winter"]  = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)
    df["is_summer"]  = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    return df

def make_target_lag_roll(df: pd.DataFrame, target: str, max_lag: int = 72, roll_list=(3,6,12,24,48)) -> pd.DataFrame:
    n = len(df)
    if n <= 10:
        max_lag = min(max_lag, max(1, n//3))
    else:
        max_lag = min(max_lag, max(3, n//5))
    if max_lag < 1:
        max_lag = 1

    #Lags
    for L in range(1, max_lag+1):
        df[f"{target}_lag{L}"] = df[target].shift(L)

    #Rolling stats & momentum
    safe_rolls = [r for r in roll_list if r <= max(2, n)]
    for R in safe_rolls:
        df[f"{target}_rmean{R}"]   = df[target].shift(1).rolling(R, min_periods=1).mean()
        df[f"{target}_rstd{R}"]    = df[target].shift(1).rolling(R, min_periods=1).std().fillna(0)
        df[f"{target}_rmin{R}"]    = df[target].shift(1).rolling(R, min_periods=1).min()
        df[f"{target}_rmax{R}"]    = df[target].shift(1).rolling(R, min_periods=1).max()
        df[f"{target}_rmedian{R}"] = df[target].shift(1).rolling(R, min_periods=1).median()
        if R >= 3:
            df[f"{target}_roc{R}"] = (df[f"{target}_rmean{R}"] -
                                      df[target].shift(R).rolling(R, min_periods=1).mean()) / (R + 1)
            df[f"{target}_momentum{R}"] = df[target].shift(1) - df[target].shift(R+1)

    #Diffs & EMAs
    df[f"{target}_diff1"]  = df[target].diff(1)
    df[f"{target}_diff24"] = df[target].diff(24)
    for alpha in [0.1, 0.3, 0.5]:
        df[f"{target}_ema_{int(alpha*10)}"] = df[target].shift(1).ewm(alpha=alpha).mean()
    return df

def build_fnn(n_in: int) -> Model:
    inputs = layers.Input(shape=(n_in,), name="tab_in")
    x = layers.Dense(1024, activation="swish")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    residual = layers.Dense(256, activation="linear")(inputs)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Add()([x, residual])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(64, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, activation="swish")(x)
    x = layers.Dropout(0.05)(x)

    out = layers.Dense(1, name="power_output")(x)
    model = Model(inputs, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

#Popup Window (with filtered "error details" tab)
def show_results_popup(target_name,
                       history,
                       y_true,
                       y_pred,
                       metrics,
                       errors_df: pd.DataFrame):
    if not TK_OK:
        print("\n[INFO] Tkinter UI not available in this environment. Skipping popup.")
        return

    root = tk.Tk()
    root.title("FNN Test Results — Metrics & Plots")
    root.geometry("1200x850")

    #Top frame: metrics + buttons
    top = ttk.Frame(root)
    top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    metrics_text = io.StringIO()
    metrics_text.write("==================== TEST METRICS (Target: Power Output) ====================\n")
    metrics_text.write(f"R²    : {metrics['r2']:.6f}\n")
    metrics_text.write(f"MSE   : {metrics['mse']:.6f}\n")
    metrics_text.write(f"RMSE  : {metrics['rmse']:.6f}\n")
    metrics_text.write(f"MAE   : {metrics['mae']:.6f}\n\n")
    metrics_text.write("----------- Percentage (normalized) -----------\n")
    metrics_text.write(f"MAE%  : {metrics['mae_pct']:.6f}%   (denom = mean(|Power Output|) = {metrics['mean_abs_y']:.6f})\n")
    metrics_text.write(f"RMSE% : {metrics['rmse_pct']:.6f}%   (denom = mean(|Power Output|) = {metrics['mean_abs_y']:.6f})\n")
    metrics_text.write(f"MSE%  : {metrics['mse_pct']:.6f}%   (denom = mean((Power Output)^2) = {metrics['mean_sq_y']:.6f})\n")
    metrics_text.write(f"R²%   : {metrics['r2_pct']:.6f}%\n")

    txt = tk.Text(top, height=10, wrap="word")
    txt.insert("1.0", metrics_text.getvalue())
    txt.config(state="disabled")
    txt.pack(side=tk.LEFT, fill=tk.X, expand=True)

    btns = ttk.Frame(top); btns.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
    def save_metrics():
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files","*.txt"), ("All files","*.*")])
        if path:
            with open(path, "w") as f:
                f.write(metrics_text.getvalue())
            messagebox.showinfo("Saved", f"Metrics saved to:\n{path}")
    def save_errors_csv():
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if path:
            errors_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Errors table saved to:\n{path}")
    ttk.Button(btns, text="Save Metrics", command=save_metrics).pack(pady=5, fill=tk.X)
    ttk.Button(btns, text="Save Errors CSV", command=save_errors_csv).pack(pady=5, fill=tk.X)
    ttk.Button(btns, text="Close", command=root.destroy).pack(pady=5, fill=tk.X)

    #Notebook with plots + table
    nb = ttk.Notebook(root); nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def add_fig_tab(fig: Figure, title: str):
        frame = ttk.Frame(nb); nb.add(frame, text=title)
        canvas = FigureCanvasTkAgg(fig, master=frame); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 1) True vs Predicted
    fig1 = Figure(figsize=(8,4), dpi=100)
    ax1 = fig1.add_subplot(111)
    ax1.plot(y_true, label="True")
    ax1.plot(y_pred, label="Pred")
    ax1.set_title(f"Test: True vs Predicted (R²={metrics['r2']:.3f}, MAE={metrics['mae']:.2f})")
    ax1.set_xlabel("Time (test index)"); ax1.set_ylabel(target_name); ax1.legend()
    fig1.tight_layout()
    add_fig_tab(fig1, "True vs Predicted")

    # 2) Parity plot
    fig2 = Figure(figsize=(5,5), dpi=100)
    ax2 = fig2.add_subplot(111)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax2.plot([lo, hi], [lo, hi], linewidth=1, color="red")
    ax2.scatter(y_pred, y_true, s=8, alpha=0.6)
    ax2.set_xlabel("Predicted Power Output"); ax2.set_ylabel("True Power Output")
    ax2.set_title(f"Pred vs True Plot (R²={metrics['r2']:.3f}, MAE={metrics['mae']:.2f})")
    fig2.tight_layout()
    add_fig_tab(fig2, "Parity Plot")

    # 3) Loss vs Epoch (Huber)
    fig3 = Figure(figsize=(6,4), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.plot(history.history.get("loss", []), label="Train Loss")
    ax3.plot(history.history.get("val_loss", []), label="Validation Loss")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss (Huber)")
    ax3.set_title("Training vs Validation Loss")
    ax3.grid(True); ax3.legend()
    fig3.tight_layout()
    add_fig_tab(fig3, "Loss vs Epoch")

    # 4) MAE vs Epoch

    if "mae" in history.history and "val_mae" in history.history:
        fig4 = Figure(figsize=(6,4), dpi=100)
        ax4 = fig4.add_subplot(111)
        ax4.plot(history.history["mae"], label="Train MAE")
        ax4.plot(history.history["val_mae"], label="Validation MAE")
        ax4.set_xlabel("Epoch"); ax4.set_ylabel("MAE")
        ax4.set_title("Training vs Validation MAE")
        ax4.grid(True); ax4.legend()
        fig4.tight_layout()
        add_fig_tab(fig4, "MAE vs Epoch")

    # 5) Filtered error table (07:00–15:00, %err < 20), with DateTime column and NO Index
    table_frame = ttk.Frame(nb)
    nb.add(table_frame, text="Error Details")

    yscroll = ttk.Scrollbar(table_frame, orient="vertical")
    xscroll = ttk.Scrollbar(table_frame, orient="horizontal")

    cols = ["DateTime","power_true","power_pred","abs_err","pct_err"]
    tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                        yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    for c in cols:
        tree.heading(c, text=c)
        w = 180 if c == "DateTime" else 140
        tree.column(c, width=w, anchor="center")

    yscroll.config(command=tree.yview)
    xscroll.config(command=tree.xview)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    xscroll.pack(side=tk.BOTTOM, fill=tk.X)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    if errors_df.empty:
        msg = ttk.Label(table_frame, text=f"No rows 07:00–15:00 with % error < {ERR_PCT_MAX:.0f} (after removing zero-power).",
                        anchor="center")
        msg.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
    else:
        for _, row in errors_df.iterrows():
            tree.insert("", "end", values=[str(row["DateTime"]),
                                           f"{row['power_true']:.6f}",
                                           f"{row['power_pred']:.6f}",
                                           f"{row['abs_err']:.6f}",
                                           f"{row['pct_err']:.6f}"])

    root.mainloop()


# Main
def main():

    
    csv_path = "data3.csv"
    
    print(f"Reading: {csv_path}")
    df0 = pd.read_csv(csv_path)

    if df0.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (datetime + target or id + target).")

    #Target is strictly the 2nd column — we’ll call it "Power Output" in UI
    target = df0.columns[1]
    print(f"Target column (2nd): {target}  [UI label: {NICE_TARGET_NAME}]")

    #DateTime handling + resample hourly
    dt_col = find_datetime_col(df0)
    if dt_col is not None:
        df0[dt_col] = pd.to_datetime(df0[dt_col], errors="coerce")
        df0 = df0.dropna(subset=[dt_col]).sort_values(dt_col)
        df = df0.set_index(dt_col)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.groupby(level=0).mean(numeric_only=True)  # aggregate duplicates
        df = df.resample("H").mean()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[target] = df[target].ffill().fillna(0)
        df = df.reset_index().rename(columns={"index":"DateTime", dt_col:"DateTime"})
    else:
        df = df0.copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=[target]).reset_index(drop=True)

    #Time features
    if "DateTime" in df.columns:
        df = add_time_features(df, "DateTime")

    #Feature engineering (lags/rolls)
    df = make_target_lag_roll(df, target, max_lag=MAX_LAG, roll_list=ROLLS)

    #Build numeric matrix & keep DateTime aligned
    num_df = df.select_dtypes(include=[np.number]).copy()
    feature_cols = [c for c in num_df.columns if c != target]
    if len(feature_cols) == 0:
        raise ValueError("No numeric features besides target were found.")

    if "DateTime" in df.columns:
        aligned = pd.concat([df[["DateTime"]], num_df[[target] + feature_cols]], axis=1)
    else:
        aligned = num_df[[target] + feature_cols].copy()
        aligned["DateTime"] = pd.NaT
        aligned = aligned[["DateTime", target] + feature_cols]

    aligned = aligned.dropna().reset_index(drop=True)

    dt_series = pd.to_datetime(aligned["DateTime"], errors="coerce")
    y = aligned[target].values.astype("float32")
    X = aligned[feature_cols].values.astype("float32")

    #time-ordered - 80/10/10
    N = len(y)
    n_test = int(math.floor(TEST_FRAC * N))
    n_val  = int(math.floor(VAL_FRAC  * N))
    n_train = N - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Split sizes invalid for N={N}. Got train={n_train}, val={n_val}, test={n_test}.")

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_te, y_te = X[n_train+n_val:], y[n_train+n_val:]
    dt_te = dt_series.iloc[n_train+n_val:].reset_index(drop=True)

    #Impute/scale
    imputer = SimpleImputer(strategy="median").fit(X_tr)
    X_tr = imputer.transform(X_tr); X_va = imputer.transform(X_va); X_te = imputer.transform(X_te)

    scalerX = MinMaxScaler(feature_range=(0, 1)).fit(X_tr)
    X_tr_s = scalerX.transform(X_tr)
    X_va_s = scalerX.transform(X_va)
    X_te_s = scalerX.transform(X_te)

    scalerY = MinMaxScaler(feature_range=(0, 1)).fit(y_tr.reshape(-1,1))
    y_tr_s = scalerY.transform(y_tr.reshape(-1,1)).ravel()
    y_va_s = scalerY.transform(y_va.reshape(-1,1)).ravel()

    #Save scalers for inference
    try:
        import pickle
        scalers = {'imputer': imputer, 'scalerX': scalerX, 'scalerY': scalerY}
        with open("scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        print("[INFO] Scalers saved for inference (scalers.pkl)")
    except Exception as e:
        print(f"[WARN] Could not save scalers: {e}")

    #Model & training
    model = build_fnn(n_in=X_tr_s.shape[1])
    ckpt = callbacks.ModelCheckpoint("best_fnn.keras", monitor="val_mae", save_best_only=True, verbose=0)
    es   = callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE, restore_best_weights=True, verbose=1)
    rlrop= callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.7, patience=max(PATIENCE//4, 3),
                                        min_lr=1e-6, verbose=1)

    history = model.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_va_s, y_va_s),
        epochs=EPOCHS, batch_size=BATCH, verbose=1,
        callbacks=[ckpt, es, rlrop],
        shuffle=True
    )

    #Save training log graph (loss vs. epoch)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(history.history.get("loss", []), label="Train Loss")
        plt.plot(history.history.get("val_loss", []), label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Huber)")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_log.png")
        plt.close()
        print("[INFO] Training log saved as training_log.png")
    except Exception as e:
        print(f"[WARN] Could not save training log plot: {e}")


    #Predict on train and test
    y_pred_tr_s = model.predict(X_tr_s, batch_size=BATCH, verbose=0).ravel()
    y_pred_tr   = scalerY.inverse_transform(y_pred_tr_s.reshape(-1,1)).ravel()
    y_pred_te_s = model.predict(X_te_s, batch_size=BATCH, verbose=0).ravel()
    y_pred_te   = scalerY.inverse_transform(y_pred_te_s.reshape(-1,1)).ravel()

    #Save predicted vs actual plots for train and test
    try:
        import matplotlib.pyplot as plt
        # Train set
        plt.figure(figsize=(10,4))
        plt.plot(y_tr, label="Train True", alpha=0.7)
        plt.plot(y_pred_tr, label="Train Pred", alpha=0.7)
        plt.title("Train Set: True vs Predicted")
        plt.xlabel("Sample")
        plt.ylabel(NICE_TARGET_NAME)
        plt.legend()
        plt.tight_layout()
        plt.savefig("train_true_vs_pred.png")
        plt.close()
        print("[INFO] Train set true vs predicted plot saved as train_true_vs_pred.png")

        # Test set
        plt.figure(figsize=(10,4))
        plt.plot(y_te, label="Test True", alpha=0.7)
        plt.plot(y_pred_te, label="Test Pred", alpha=0.7)
        plt.title("Test Set: True vs Predicted")
        plt.xlabel("Sample")
        plt.ylabel(NICE_TARGET_NAME)
        plt.legend()
        plt.tight_layout()
        plt.savefig("test_true_vs_pred.png")
        plt.close()
        print("[INFO] Test set true vs predicted plot saved as test_true_vs_pred.png")
    except Exception as e:
        print(f"[WARN] Could not save true vs predicted plots: {e}")

    # Metrics
    mae  = mean_absolute_error(y_te, y_pred_te)
    mse  = mean_squared_error(y_te, y_pred_te)
    rmse = math.sqrt(mse)
    r2   = r2_score(y_te, y_pred_te)

    eps = 1e-12
    mean_abs_y = float(np.mean(np.abs(y_te)))
    mean_sq_y  = float(np.mean(y_te.astype("float64")**2))
    mae_pct  = 100.0 * mae  / max(mean_abs_y, eps)
    rmse_pct = 100.0 * rmse / max(mean_abs_y, eps)
    mse_pct  = 100.0 * mse  / max(mean_sq_y, eps)
    r2_pct   = 100.0 * r2

    print("\n==================== TEST METRICS (Target: Power Output) ====================")
    print(f"R²    : {r2:8.4f}")
    print(f"MSE   : {mse:10.4f}")
    print(f"RMSE  : {rmse:10.4f}")
    print(f"MAE   : {mae:10.4f}")
    print("\n----------- Percentage (normalized) -----------")
    print(f"MAE%  : {mae_pct:10.4f}%   (denom = mean(|Power Output|) = {mean_abs_y:.6f})")
    print(f"RMSE% : {rmse_pct:10.4f}%   (denom = mean(|Power Output|) = {mean_abs_y:.6f})")
    print(f"MSE%  : {mse_pct:10.4f}%   (denom = mean((Power Output)^2) = {mean_sq_y:.6f})")
    print(f"R²%   : {r2_pct:10.4f}%")

    # Save predictions (+ DateTime)
    out = pd.DataFrame({"DateTime": dt_te, "power_true": y_te, "power_pred": y_pred_te})
    out.to_csv("solar_predictions_test.csv", index=False)
    with open("test_metrics.txt","w") as f:
        f.write(
            f"R2={r2:.6f}\nMSE={mse:.6f}\nRMSE={rmse:.6f}\nMAE={mae:.6f}\n"
            f"MAE%={mae_pct:.6f}%\nRMSE%={rmse_pct:.6f}%\nMSE%={mse_pct:.6f}%\nR2%={r2_pct:.6f}%\n"
            f"mean_abs_power={mean_abs_y:.6f}\nmean_sq_power={mean_sq_y:.6f}\n"
        )
    #Build error table: remove zero-power, filter hours 07–15 and %err < 20
    abs_err = np.abs(y_te - y_pred_te)

    errors_df = pd.DataFrame({
        "DateTime": pd.to_datetime(dt_te),
        "power_true": y_te.astype(float),
        "power_pred": y_pred_te.astype(float),
        "abs_err": abs_err.astype(float)
    })

    #Remove night-time / near-zero target
    errors_df = errors_df[errors_df["power_true"] > ZERO_TOL].copy()

    #Hour filter 07:00–15:00 (inclusive)
    if errors_df["DateTime"].notna().any():
        hrs = errors_df["DateTime"].dt.hour
        errors_df = errors_df[hrs.between(HOUR_START, HOUR_END)].copy()
    else:
        print(f"[WARN] No valid DateTime available; cannot filter by hours {HOUR_START:02d}:00–{HOUR_END:02d}:00.")

    # % error and strict threshold (< 20)
    errors_df["pct_err"] = 100.0 * errors_df["abs_err"] / errors_df["power_true"].abs()
    errors_df = errors_df[errors_df["pct_err"] < ERR_PCT_MAX].copy()

    #Sort & save filtered table (no Index column)
    errors_df = errors_df.sort_values("pct_err", ascending=True).reset_index(drop=True)
    errors_df.to_csv("solar_test_errors.csv", index=False)

    #POPUP WINDOW with filtered table
    metrics_pack = {
        "r2": r2, "mse": mse, "rmse": rmse, "mae": mae,
        "mae_pct": mae_pct, "rmse_pct": rmse_pct, "mse_pct": mse_pct, "r2_pct": r2_pct,
        "mean_abs_y": mean_abs_y, "mean_sq_y": mean_sq_y
    }
    show_results_popup(NICE_TARGET_NAME, history, y_te, y_pred_te, metrics_pack, errors_df)

if __name__ == "__main__":
    main()

