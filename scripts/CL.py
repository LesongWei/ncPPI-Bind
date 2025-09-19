
import torch

def supcon_loss(z, y, temperature=0.07, ignore_index=-1):
    """
    z: [N, D]   已经来自投影头、且会再做一次 L2 norm
    y: [N]      标签 {0,1} 或 {-1,0,1} ；-1 会被忽略
    """
    # 有效索引
    valid = (y != ignore_index)
    z = z[valid]
    y = y[valid]

    N = z.size(0)
    if N <= 1:
        return z.new_tensor(0.0)

    # 归一化
    z = F.normalize(z, dim=-1)

    # 相似度
    sim = torch.matmul(z, z.t()) / temperature  # [N, N]

    # 构造“同类为正”的 mask（排除对角）
    labels = y.view(-1, 1)
    pos_mask = (labels == labels.t()).float()
    self_mask = torch.eye(N, device=z.device)
    pos_mask = pos_mask - self_mask  # 对角线置0

    # log-softmax over rows
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # 每个样本的正样本平均 log_prob（若无正样本，则该样本不计入）
    pos_counts = pos_mask.sum(dim=1)  # [N]
    valid_pos = pos_counts > 0
    if valid_pos.sum() == 0:
        return z.new_tensor(0.0)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp_min(1.0)
    loss = -(mean_log_prob_pos[valid_pos]).mean()
    return loss


def supcon_loss_pos_only(z, y, temperature=0.07, ignore_index=-1):
    """
    只把 label==1 之间当作正对（更贴位点任务）。
    其余均为负样本或忽略；仍然全体参与分母（对比学习的负样本）。
    """
    valid = (y != ignore_index)
    z = z[valid]
    y = y[valid]

    N = z.size(0)
    if N <= 1:
        return z.new_tensor(0.0)

    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.t()) / temperature

    labels = y.view(-1, 1)
    pos_mask = ((labels == 1) & (labels.t() == 1)).float()
    self_mask = torch.eye(N, device=z.device)
    pos_mask = pos_mask - (pos_mask * self_mask)  # 去掉对角上那一格

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_counts = pos_mask.sum(dim=1)
    valid_pos = pos_counts > 0
    if valid_pos.sum() == 0:
        return z.new_tensor(0.0)

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_counts.clamp_min(1.0)
    return -(mean_log_prob_pos[valid_pos]).mean()

def supcon_with_queue(z, y, queue, temperature=0.07, pos_only=False, ignore_index=-1):
    """
    z: [N, D] 当前batch（序列内）残基投影
    y: [N]
    queue: 提供历史的 (z_mem, y_mem)
    """
    z_list = [z]
    y_list = [y]
    if queue is not None:
        z_mem, y_mem = queue.get()  # 可能返回 None
        if z_mem is not None and y_mem is not None and z_mem.numel() > 0:
            z_list.append(z_mem.to(z.device))
            y_list.append(y_mem.to(y.device))

    z_all = torch.cat(z_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    if pos_only:
        loss = supcon_loss_pos_only(z_all, y_all, temperature=temperature, ignore_index=ignore_index)
    else:
        loss = supcon_loss(z_all, y_all, temperature=temperature, ignore_index=ignore_index)

    # 入队（**detach**，避免梯度堆积）
    if queue is not None:
        queue.enqueue(z.detach(), y.detach())

    return loss

class MemoryQueue:
    def __init__(self, dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.feats = torch.empty(0, dim, device=device)
        self.labels = torch.empty(0, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, feats, labels):
        # feats: [N, dim] (已detach并在CPU/GPU与self一致)
        self.feats = torch.cat([self.feats, feats], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        if self.feats.size(0) > self.max_size:
            overflow = self.feats.size(0) - self.max_size
            self.feats = self.feats[overflow:]
            self.labels = self.labels[overflow:]

    def get(self):
        return self.feats, self.labels