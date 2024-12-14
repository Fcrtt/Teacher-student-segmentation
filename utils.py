import torch
import torch.nn as nn
import torch.nn.functional as F


class ESPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ESPBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilation_rates
        ])
        total_channels = len(dilation_rates) * out_channels
        self.bn = nn.BatchNorm2d(total_channels)

    def forward(self, x):
        x = self.conv1x1(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.bn(x)
        return F.relu(x)



class ESPNet(nn.Module):
    def __init__(self, num_class):
        super(ESPNet, self).__init__()
        self.initial = nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.esp_block1 = ESPBlock(64, 128, dilation_rates=[1, 2, 4])
        self.esp_block2 = ESPBlock(384, 128, dilation_rates=[1, 2, 4, 8])
        self.btn = nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False)
        self.conv_out = nn.Conv2d(256, num_class, kernel_size=1, stride=1)

    def forward(self, x):
        original_size = x.shape[-2:]  # Save original input spatial dimensions
        x = self.relu(self.bn1(self.initial(x)))
        x = self.esp_block1(x)
        x = self.esp_block2(x)
        x = self.btn(x)
        x = self.conv_out(x)
        # Ensure the output size matches the original input size
        return F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

def distillation_loss(student_logits, teacher_logits, temperature=3):
    # Softmax avec température
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.softmax(student_logits / temperature, dim=1)
    # KL-Divergence
    return F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (temperature ** 2)

# Combinaison des pertes
def combined_loss(student_logits, teacher_logits, labels, alpha=0.5, beta=0.5, temperature=3):
    # Ensure labels are in the correct shape and type
    labels = labels.squeeze(1).long()  # Remove channel dimension and ensure Long type
    # Supervised loss
    supervised_loss = F.cross_entropy(student_logits, labels)
    # Distillation loss
    distill_loss = distillation_loss(student_logits, teacher_logits, temperature)
    # Combined loss
    return alpha * supervised_loss + beta * distill_loss


