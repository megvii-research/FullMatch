import math
import megengine as mge
import megengine.module as M
import megengine.functional as F
import megengine.hub as hub


momentum = 0.999


class BasicBlock(M.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False, momentum=0.999, sync_bn=False):
        super(BasicBlock, self).__init__()
        if sync_bn:
            self.bn1 = M.SyncBatchNorm(in_planes, momentum=momentum, eps=0.001)
        else:
            self.bn1 = M.BatchNorm2d(in_planes, momentum=momentum, eps=0.001)
        self.relu1 = M.LeakyReLU(negative_slope=0.1)
        self.conv1 = M.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        if sync_bn:
            self.bn2 = M.SyncBatchNorm(out_planes, momentum=momentum, eps=0.001)
        else:
            self.bn2 = M.BatchNorm2d(out_planes, momentum=momentum, eps=0.001)
        self.relu2 = M.LeakyReLU(negative_slope=0.1)
        self.conv2 = M.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and M.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return F.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(M.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False, momentum=0.999, sync_bn=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, momentum, sync_bn)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual, momentum, sync_bn):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual, momentum=momentum, sync_bn=sync_bn))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(M.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False, bn_momentum=0.999, sync_bn=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = M.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True, momentum=bn_momentum, sync_bn=sync_bn)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate, momentum=bn_momentum, sync_bn=sync_bn)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate, momentum=bn_momentum, sync_bn=sync_bn)
        # global average pooling and classifier
        if sync_bn:
            self.bn1 = M.SyncBatchNorm(channels[3], momentum=momentum, eps=0.001)
        else:
            self.bn1 = M.BatchNorm2d(channels[3], momentum=bn_momentum, eps=0.001)
        self.relu = M.LeakyReLU(negative_slope=0.1)
        self.fc = M.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = M.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, M.BatchNorm2d) or isinstance(m, M.SyncBatchNorm):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.xavier_normal_(m.weight)
                M.init.zeros_(m.bias)

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.reshape(-1, self.channels)
        output = self.fc(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                return output


class build_WideResNet:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.999, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False, sync_bn=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix
        self.sync_bn = sync_bn

    def build(self, num_classes):
        return WideResNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
            bn_momentum=self.bn_momentum,
            sync_bn = self.sync_bn,
        )


if __name__ == '__main__':
    wrn_builder = build_WideResNet(1, 10, 2, 0.999, 0.1, 0.0)
    wrn = wrn_builder.build(10)
    print(wrn)
