import math

def calculate_final_score(bpp: float, acc_jpeg: float, acc_awgn: float, acc_mblur: float, niqe: float) -> float:
    """
    严格按照 PPT 里的高清公式计算最终得分
    """
    # 1. 容量得分 (30分): 30 * min(1, ln(1 + 1000*bpp) / ln(1001))
    raw_capacity = math.log(1 + 1000 * bpp) / math.log(1001) if bpp > 0 else 0.0
    score_capacity = 30 * min(1.0, raw_capacity)
        
    # 2. 鲁棒性得分 (50分): 50 * (1/3 * sum(max(0, 2*Acc - 1)))
    # 注意：如果 Acc <= 0.5，该项得分为 0
    robust_jpeg = max(0.0, 2 * acc_jpeg - 1)
    robust_awgn = max(0.0, 2 * acc_awgn - 1)
    robust_mblur = max(0.0, 2 * acc_mblur - 1)
    
    avg_robustness = (robust_jpeg + robust_awgn + robust_mblur) / 3.0
    score_robustness = 50 * avg_robustness
    
    # 3. 安全性得分 (20分): 20 * min(1, max(0, (8 - NIQE) / 5))
    niqe_normalized = (8 - niqe) / 5.0
    niqe_normalized = max(0.0, min(1.0, niqe_normalized)) # 限制在 0-1 之间
    score_security = 20 * niqe_normalized
    
    total_score = score_capacity + score_robustness + score_security
    
    print(f"\n===== 得分详情 =====")
    print(f"嵌入容量 bpp: {bpp:.4f} (原始计算值 {raw_capacity:.4f}, 得分: {score_capacity:.2f}/30)")
    print(f"鲁棒性得分: JPEG(Acc={acc_jpeg:.2f} -> {robust_jpeg:.2f}) | AWGN(Acc={acc_awgn:.2f} -> {robust_awgn:.2f}) | MBLUR(Acc={acc_mblur:.2f} -> {robust_mblur:.2f})")
    print(f"鲁棒性综合得分: {score_robustness:.2f}/50")
    print(f"安全性 NIQE: {niqe:.4f} (归一化 {niqe_normalized:.2f}, 得分: {score_security:.2f}/20)")
    print(f"--------------------")
    print(f"最终总分: {total_score:.2f} / 100")
    
    return total_score
