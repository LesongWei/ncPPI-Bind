import torch


class ClassCountEMA:
    def __init__(self, decay=0.9, eps=1.0):
        self.decay = decay
        self.eps = eps
        self.pos = 0.0
        self.neg = 0.0
        self.inited = False

    def update(self, target):
        valid = (target != -1)
        if valid.sum() == 0:
            return
        y = target[valid]
        pos = float((y == 1).sum().item())
        neg = float((y == 0).sum().item())
        if not self.inited:
            self.pos, self.neg, self.inited = pos, neg, True
        else:
            self.pos = self.decay * self.pos + (1 - self.decay) * pos
            self.neg = self.decay * self.neg + (1 - self.decay) * neg

    def weights(self, task_type="protein", min_w=1.0, max_w=8.0, normalize=True, device=None):
        pos_f = self.pos + self.eps
        neg_f = self.neg + self.eps
        if task_type == "protein":
            w0, w1 = 1.0, (neg_f / pos_f)
        else:
            w0, w1 = (pos_f / neg_f), 1.0
        w0 = max(min_w, min(max_w, w0))
        w1 = max(min_w, min(max_w, w1))
        w = torch.tensor([w0, w1], dtype=torch.float)
        if normalize:
            w = 2.0 * w / (w.sum().clamp_min(1e-8))
        return w.to(device) if device else w
