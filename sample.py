import torch
import torch.nn as nn


class SampleAdaptor:
    def __init__(self, flops_budget, config_file="config/sample/relu/default.json", **kwargs):
        self.flops_budget = flops_budget
        self.relu = nn.ReLU()
        if config_file and flops_budget != -1:
            self.adaptive = True
            self.parse_config(config_file)
        else:
            self.adaptive = False

    def parse_config(self, file):
        import json
        with open(file, "r") as load_f:
            load_dict = json.load(load_f)
        self.type = load_dict["type"]
        self.detach = bool(load_dict["detach"])
        self.multiply = load_dict["multiply"]
        self.add = load_dict["add"]
        self.over = load_dict["over"]
        self.under = load_dict["under"]

    def extract_flops(self, flops_real, flops_mask, flops_ori, batch_size):
        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        flops_backbone = flops_conv1.repeat(batch_size) + flops_fc.repeat(batch_size) + torch.sum(flops_tensor, (1))
        flops_mask = flops_mask.sum().repeat(batch_size)
        flops_ori = flops_ori.sum() + flops_conv1 + flops_fc
        flops_real = flops_mask + flops_backbone
        return flops_real, flops_ori

    def extract_meta(self, outputs, targets):
        pos = torch.softmax(outputs, dim=1)
        preds = torch.max(outputs, dim=1)[1]
        target_pos = []
        for p, target in zip(pos, targets):
            target_pos.append(p[target])
        return torch.max(pos, dim=1)[0], preds, torch.Tensor(target_pos).cuda()

    def update(self, outputs, targets, flops_real, flops_mask, flops_ori, batch_size):
        if not self.adaptive:
            return None
        self.max_FLOPs, flops_ratio = self.extract_flops(flops_real, flops_mask, flops_ori, batch_size)
        if self.type == "relu":
            possibs, preds, target_pos = self.extract_meta(outputs, targets)
            conf_dist = possibs - target_pos
            alpha = self.over * self.relu(target_pos + flops_ratio - 0.5 - self.flops_budget) ** 2 - \
                    self.under * conf_dist * self.relu(conf_dist + flops_ratio - 0.5 - self.flops_budget)
            weights = (torch.exp(self.multiply * alpha) + self.add)
        else:
            raise NotImplementedError
        return weights.detach() if self.detach else weights
