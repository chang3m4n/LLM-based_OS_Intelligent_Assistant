import json
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
import numpy as np  

# trainer_state.json文件的实际路径
TRAINER_STATE_PATH = "./finetune_results/checkpoint-225/trainer_state.json"
# ------------------------------------------------------------------------

def load_loss_data(state_file_path):
    """读取训练日志数据"""
    if not os.path.exists(state_file_path):
        raise FileNotFoundError(f"文件不存在，请检查路径：{state_file_path}")
    
    with open(state_file_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)
    
    log_history = state_data.get("log_history", [])
    if not log_history:
        raise ValueError("日志数据为空，可能文件已损坏或训练未完成")
    
    return log_history

def extract_train_eval_loss(log_history):
    """提取并对齐训练和评估Loss数据"""
    # 提取所有评估记录（带轮次和评估Loss）
    eval_records = []
    for log in log_history:
        if "eval_loss" in log and "epoch" in log:
            eval_records.append({
                "epoch": log["epoch"],
                "eval_loss": log["eval_loss"]
            })
    
    # 提取所有训练记录（带轮次和训练Loss）
    train_records = []
    for log in log_history:
        if "loss" in log and "epoch" in log and "eval_loss" not in log:
            train_records.append({
                "epoch": log["epoch"],
                "train_loss": log["loss"]
            })
    
    # 确保训练记录不为空
    if not train_records:
        raise ValueError("未找到任何训练Loss数据")
    
    # 为每个评估轮次精确匹配最接近的训练Loss
    matched_train_losses = []
    for eval_item in eval_records:
        eval_epoch = eval_item["epoch"]
        # 找到最接近当前评估轮次的训练记录
        closest_diff = float('inf')
        closest_train_loss = None
        
        for train_item in train_records:
            diff = abs(train_item["epoch"] - eval_epoch)
            if diff < closest_diff:
                closest_diff = diff
                closest_train_loss = train_item["train_loss"]
        
        # 确保一定能找到训练Loss
        if closest_train_loss is None:
            closest_train_loss = train_records[-1]["train_loss"]
            
        matched_train_losses.append(closest_train_loss)
    
    # 严格确保维度完全一致
    epochs = [item["epoch"] for item in eval_records]
    eval_losses = [item["eval_loss"] for item in eval_records]
    train_losses = matched_train_losses[:len(epochs)] 
    
    # 最终检查
    if len(train_losses) != len(eval_losses) or len(epochs) != len(eval_losses):
        raise ValueError(f"数据维度仍不匹配: 训练Loss({len(train_losses)}) | 评估Loss({len(eval_losses)}) | 轮次({len(epochs)})")
    
    return train_losses, eval_losses, epochs

def plot_and_save_loss(train_losses, eval_losses, epochs):

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    plt.figure(figsize=(12, 6))

    epochs_np = np.array(epochs)
    train_losses_np = np.array(train_losses)
    eval_losses_np = np.array(eval_losses)
    epochs_smooth = np.linspace(epochs_np.min(), epochs_np.max(), 300)  # 300个密集点

    # 3. 用三次样条插值生成平滑曲线
    if len(epochs_np) >= 3: 
        train_spline = make_interp_spline(epochs_np, train_losses_np, k=3)  # k=3为三次样条
        eval_spline = make_interp_spline(epochs_np, eval_losses_np, k=3)
        train_losses_smooth = train_spline(epochs_smooth)
        eval_losses_smooth = eval_spline(epochs_smooth)
    else:  
        epochs_smooth = epochs_np
        train_losses_smooth = train_losses_np
        eval_losses_smooth = eval_losses_np

    # -------------------------- 绘制平滑曲线 --------------------------
    # 训练Loss曲线
    plt.plot(epochs_smooth, train_losses_smooth, label="Train Loss",
             color="#2E86AB", linewidth=3, alpha=0.8) 
    # 评估Loss曲线
    plt.plot(epochs_smooth, eval_losses_smooth, label="Eval Loss",
             color="#F18F01", linewidth=3, alpha=0.8)

    # -------------------------- 图表细节优化 --------------------------
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.title("Qwen2.5-7B-Instruct Training & Evaluation Loss", fontsize=16, pad=20)
    plt.legend(fontsize=12, frameon=True, shadow=True)
    plt.grid(alpha=0.3, linestyle="--", linewidth=1) 
    # 保存图片
    save_dir = os.path.dirname(TRAINER_STATE_PATH)
    save_path = os.path.join(save_dir, "loss_curve_smoothed.png")
    plt.tight_layout()  
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"曲线图已保存至：{save_path}")
    print(f"数据维度：原始训练Loss({len(train_losses)}) | 原始评估Loss({len(eval_losses)})")

if __name__ == "__main__":
    try:
        log_data = load_loss_data(TRAINER_STATE_PATH)
        train_loss, eval_loss, epoch_list = extract_train_eval_loss(log_data)
        plot_and_save_loss(train_loss, eval_loss, epoch_list)
    except ImportError:
        print("❌ 缺少依赖库，请先执行：pip install scipy numpy")
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")