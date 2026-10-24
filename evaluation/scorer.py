import math

def calculate_final_score(bpp: float, acc_jpeg: float, acc_awgn: float, acc_mblur: float, niqe: float) -> float:
    """
    严格按照 PPT 里的公式计算最终得分
    """
    # 1. 容量得分 (30分): 30 * (ln(1 + 1000*bpp) / ln(1001))
    # 假设 bpp 达到 1 时，该项得分为 30
    if bpp > 0:
        score_capacity = 30 * (math.log(1 + 1000 * bpp) / math.log(1001))
    else:
        score_capacity = 0.0
        
    # 2. 鲁棒性得分 (50分): 50 * (1/3 * sum(Acc))
    avg_acc = (acc_jpeg + acc_awgn + acc_mblur) / 3.0
    score_robustness = 50 * avg_acc
    
    # 3. 安全性得分 (20分): 20 * min(max(0, (8 - NIQE) / 5), 1)
    # 截断归一化
    niqe_normalized = (8 - niqe) / 5.0
    niqe_normalized = max(0.0, min(1.0, niqe_normalized)) # 限制在 0-1 之间
    score_security = 20 * niqe_normalized
    
    total_score = score_capacity + score_robustness + score_security
    
    print(f"\n===== 得分详情 =====")
    print(f"嵌入容量 bpp: {bpp:.4f} (得分: {score_capacity:.2f}/30)")
    print(f"鲁棒性 Acc (JPEG/AWGN/MBLUR): {acc_jpeg:.4f}/{acc_awgn:.4f}/{acc_mblur:.4f} (平均得分: {score_robustness:.2f}/50)")
    print(f"安全性 NIQE: {niqe:.4f} (得分: {score_security:.2f}/20)")
    print(f"--------------------")
    print(f"最终总分: {total_score:.2f} / 100")
    
    return total_score