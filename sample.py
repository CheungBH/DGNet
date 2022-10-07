import torch
import torch.nn as nn


class SampleAdaptor:
    def __init__(self, flops_budget, config_file="config/sample/relu/default.json", num_epochs=100, **kwargs):
        self.flops_budget = flops_budget
        self.relu = nn.ReLU()
        self.num_epochs = num_epochs
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
        self.over = load_dict["over"]
        self.under = load_dict["under"]
        self.begin_epoch = self.num_epochs * load_dict["begin_ratio"]
        self.flops_ratio_range = load_dict["flops_range"]
        if self.type == "relu":
            self.multiply = load_dict["multiply"]
            self.add = load_dict["add"]
        elif self.type == "distribute":
            over_param = load_dict["over_param"]
            self.over_A, self.over_B, self.over_C, self.over_D = \
                over_param["over_A"], over_param["over_B"], over_param["over_C"], over_param["over_D"]
            under_param = load_dict["under_param"]
            self.under_A, self.under_B, self.under_C, self.under_D, self.under_E, self.under_F \
                = under_param["under_A"], under_param["under_B"], under_param["under_C"], under_param["under_D"], \
                  under_param["under_E"], under_param["under_F"]

    def extract_flops(self, flops_real, flops_mask, flops_ori, batch_size):
        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        flops_backbone = flops_conv1.repeat(batch_size) + flops_fc.repeat(batch_size) + torch.sum(flops_tensor, (1))
        flops_mask = flops_mask.sum().repeat(batch_size)
        flops_ori = flops_ori.sum() + flops_conv1 + flops_fc
        flops_real = flops_mask + flops_backbone
        return flops_real, flops_real/flops_ori

    def extract_meta(self, outputs, targets):
        pos = torch.softmax(outputs, dim=1)
        preds = torch.max(outputs, dim=1)[1]
        target_pos = []
        for p, target in zip(pos, targets):
            target_pos.append(p[target])
        return torch.max(pos, dim=1)[0], preds, torch.Tensor(target_pos).cuda()

    def flops_dist_limit(self, dists):
        dists = self.relu(dists - self.flops_ratio_range) + self.flops_ratio_range
        dists = self.relu(dists + self.flops_ratio_range) - self.flops_ratio_range
        return dists

    def update(self, outputs, targets, flops_real, flops_mask, flops_ori, batch_size, epoch):
        if not self.adaptive:
            return None

        if epoch < self.begin_epoch:
            return torch.tensor([torch.tensor(1) for _ in range(outputs.shape[0])]).cuda()

        self.max_FLOPs, flops_ratio = self.extract_flops(flops_real, flops_mask, flops_ori, batch_size)
        possibs, preds, target_pos = self.extract_meta(outputs, targets)
        conf_dist = possibs - target_pos
        flops_dist = flops_ratio - self.flops_budget
        if self.flops_ratio_range > 0:
            flops_dist = self.flops_dist_limit(flops_dist)
        if self.type == "relu":
            alpha = self.over * flops_dist ** 2 * self.relu(target_pos - 0.5) ** 2 - \
                    self.under * conf_dist * self.relu(0.5 - conf_dist) ** 2 * flops_dist ** 2
            weights = torch.exp(self.multiply * alpha) + self.add
        elif self.type == "distribute":
            correct_dist = self.over_A * possibs ** 2 + 2 * self.over_B * possibs * flops_dist - self.over_B * flops_dist + \
                           self.over_C * possibs + self.over_D
            wrong_dist = self.under_A * conf_dist ** 2 + 2 * self.under_B * conf_dist * flops_dist + self.under_C * flops_dist ** 2 \
                         + self.under_D * conf_dist + self.under_E * flops_dist + self.under_F
            whole_dist = correct_dist * (conf_dist == 0).int() * self.over + \
                         wrong_dist * (conf_dist != 0).int() * self.under
            weights = torch.exp(whole_dist)
        else:
            raise NotImplementedError
        return weights.detach() if self.detach else weights
