import math

class EMA:
    """简洁稳定的滑动均值，用于平滑 loss 比例估计。"""
    def __init__(self, decay=0.9):
        self.decay = decay
        self.buf = {}

    def update(self, key, val):
        if key not in self.buf:
            self.buf[key] = float(val)
        else:
            self.buf[key] = self.decay * self.buf[key] + (1 - self.decay) * float(val)

    def get(self, key, default=0.0):
        return self.buf.get(key, default)

def linear_warmup_scale(global_step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, (global_step + 1) / float(warmup_steps)))

def cosine_warmup_scale(global_step, warmup_steps):
    # 更平滑的 warmup（可选）
    if warmup_steps <= 0:
        return 1.0
    s = min(1.0, (global_step + 1) / float(warmup_steps))
    return 0.5 * (1 - math.cos(math.pi * s))

class AdaptiveLambda:
    """
    自适应 λ：
    λ = base * warmup_scale * balance_scale
    其中 balance_scale 让对比损失与监督损失保持可控比例。
    """
    def __init__(
        self,
        base_prot=0.2, base_pep=0.1,
        warmup_steps=300,                   # 你的 5920/batch_size=1，建议 300~600
        ema_decay=0.9,
        target_ratio=0.3,                  # 希望 “contrast / supervised ≈ 0.3”
        beta=0.5,                          # 平衡强度（0.3~1.0 值越大调整越猛）
        min_lambda=0.02, max_lambda=2.0,   # 保护上下限
        use_cosine_warmup=False
    ):
        self.base_prot = base_prot
        self.base_pep  = base_pep
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.target_ratio = target_ratio
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.ema = EMA(decay=ema_decay)
        self.global_step = 0
        self.warmup_fn = cosine_warmup_scale if use_cosine_warmup else linear_warmup_scale

    def step(self, prot_sup, pep_sup, prot_con, pep_con):
        """
        prot_sup: 监督（交叉熵）蛋白质损失（标量 Tensor 或 float）
        pep_sup : 监督（交叉熵）多肽损失
        prot_con: 对比损失（蛋白质）
        pep_con : 对比损失（多肽）
        返回 (lambda_prot, lambda_pep)
        """
        # --- 更新 EMA ---
        self.ema.update("prot_sup", float(prot_sup))
        self.ema.update("pep_sup",  float(pep_sup))
        self.ema.update("prot_con", float(prot_con))
        self.ema.update("pep_con",  float(pep_con))

        # --- 计算 warmup ---
        warm = self.warmup_fn(self.global_step, self.warmup_steps)

        # --- 计算平衡缩放：让 contrast/supervised 接近 target_ratio ---
        eps = 1e-8
        # 当前比例（用 EMA 平滑）
        r_prot = self.ema.get("prot_con") / (self.ema.get("prot_sup") + eps)  # 实际 contrast/sup
        r_pep  = self.ema.get("pep_con")  / (self.ema.get("pep_sup")  + eps)

        # 希望 r ≈ target_ratio -> 缩放因子 = (target_ratio / r)^beta
        # 若对比 > 目标，则 scale < 1 降低 λ；反之提高 λ
        scale_prot = (self.target_ratio / max(r_prot, eps)) ** self.beta
        scale_pep  = (self.target_ratio / max(r_pep,  eps)) ** self.beta

        # --- 组合得到最终 λ，并裁剪 ---
        lam_prot = self.base_prot * warm * scale_prot
        lam_pep  = self.base_pep  * warm * scale_pep

        lam_prot = float(max(self.min_lambda, min(self.max_lambda, lam_prot)))
        lam_pep  = float(max(self.min_lambda, min(self.max_lambda, lam_pep)))

        self.global_step += 1
        return lam_prot, lam_pep