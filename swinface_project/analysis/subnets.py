
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .cbam import ChannelGate, SpatialGate


class ConvLayer(torch.nn.Module):

    def __init__(self, in_chans=768, out_chans=512, conv_mode="normal", kernel_size=3):
        super().__init__()
        self.conv_mode = conv_mode

        if conv_mode == "normal":
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        elif conv_mode == "split":
            self.convs = nn.ModuleList()
            for j in range(len(in_chans)):
                conv = nn.Conv2d(in_chans[j], out_chans[j], kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
                self.convs.append(conv)

            self.cut = [0 for i in range(len(in_chans)+1)]
            self.cut[0] = 0
            for i in range(1, len(in_chans)+1):
                self.cut[i] = self.cut[i - 1] + in_chans[i-1]

    def forward(self, x):
        if self.conv_mode == "normal":
            x = self.conv(x)

        elif self.conv_mode == "split":
            outputs = []
            for j in range(len(self.cut)-1):
                input_map = x[:, self.cut[j]:self.cut[j+1]]
                #print(input_map.shape)
                output_map = self.convs[j](input_map)
                outputs.append(output_map)
                #print(output_map.shape)
            x = torch.cat(outputs, dim=1)

        return x


class LANet(torch.nn.Module):
    def __init__(self, in_chans=512, reduction_ratio=2.0):
        super().__init__()

        self.in_chans = in_chans
        self.mid_chans = int(self.in_chans/reduction_ratio)

        self.conv1 = nn.Conv2d(self.in_chans, self.mid_chans, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(self.mid_chans, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))

        return x


def MAD(x, p=0.6):
    B, C, W, H = x.shape

    mask1 = torch.cat([torch.randperm(C).unsqueeze(dim=0) for j in range(B)], dim=0).cuda()
    mask2 = torch.rand([B, C]).cuda()
    ones = torch.ones([B, C], dtype=torch.float).cuda()
    zeros = torch.zeros([B, C], dtype=torch.float).cuda()
    mask = torch.where(mask1 == 0, zeros, ones)
    mask = torch.where(mask2 < p, mask, ones)

    x = x.permute(2, 3, 0, 1)
    x = x.mul(mask)
    x = x.permute(2, 3, 0, 1)
    return x


class LANets(torch.nn.Module):

    def __init__(self, branch_num=2, feature_dim=512, la_reduction_ratio=2.0, MAD=MAD):
        super().__init__()

        self.LANets = nn.ModuleList()
        for i in range(branch_num):
            self.LANets.append(LANet(in_chans=feature_dim, reduction_ratio=la_reduction_ratio))

        self.MAD = MAD
        self.branch_num = branch_num

    def forward(self, x):

        B, C, W, H = x.shape

        outputs = []
        for lanet in self.LANets:
            output = lanet(x)
            outputs.append(output)

        LANets_output = torch.cat(outputs, dim=1)

        if self.MAD and self.branch_num != 1:
            LANets_output = self.MAD(LANets_output)

        mask = torch.max(LANets_output, dim=1).values.reshape(B, 1, W, H)
        x = x.mul(mask)

        return x


class FeatureAttentionNet(torch.nn.Module):
    def __init__(self, in_chans=768, feature_dim=512, kernel_size=3,
                 conv_shared=False, conv_mode="normal",
                 channel_attention=None, spatial_attention=None,
                 pooling="max", la_branch_num=2):
        super().__init__()

        self.conv_shared = conv_shared
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

        if not self.conv_shared:
            if conv_mode == "normal":
                self.conv = ConvLayer(in_chans=in_chans, out_chans=feature_dim,
                                      conv_mode="normal", kernel_size=kernel_size)
            elif conv_mode == "split" and in_chans == 2112:
                self.conv = ConvLayer(in_chans=[192, 384, 768, 768], out_chans=[47, 93, 186, 186],
                                      conv_mode="split", kernel_size=kernel_size)

        if self.channel_attention == "CBAM":
            self.channel_attention = ChannelGate(gate_channels=feature_dim)

        if self.spatial_attention == "CBAM":
            self.spatial_attention = SpatialGate()
        elif self.spatial_attention == "LANet":
            self.spatial_attention = LANets(branch_num=la_branch_num, feature_dim=feature_dim)

        if pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(num_features=feature_dim, eps=2e-5)

    def forward(self, x):

        if not self.conv_shared:
            x = self.conv(x)

        if self.channel_attention:
            x = self.channel_attention(x)

        if self.spatial_attention:
            x = self.spatial_attention(x)

        x = self.act(x)
        B, C, _, __ = x.shape
        x = self.pool(x).reshape(B, C)
        x = self.norm(x)

        return x


class FeatureAttentionModule(torch.nn.Module):
    def __init__(self, branch_num=11, in_chans=2112, feature_dim=512, conv_shared=False, conv_mode="split", kernel_size=3,
                 channel_attention="CBAM", spatial_attention=None, la_num_list=[2 for j in range(11)], pooling="max"):
        super().__init__()


        self.conv_shared = conv_shared
        if self.conv_shared:
            if conv_mode == "normal":
                self.conv = ConvLayer(in_chans=in_chans, out_chans=feature_dim,
                                      conv_mode="normal", kernel_size=kernel_size)
            elif conv_mode == "split" and in_chans == 2112:
                self.conv = ConvLayer(in_chans=[192, 384, 768, 768], out_chans=[47, 93, 186, 186],
                                      conv_mode="split", kernel_size=kernel_size)

        self.nets = nn.ModuleList()
        for i in range(branch_num):
            net = FeatureAttentionNet(in_chans=in_chans, feature_dim=feature_dim,
                                      conv_shared=conv_shared, conv_mode=conv_mode, kernel_size=kernel_size,
                                      channel_attention=channel_attention, spatial_attention=spatial_attention,
                                      la_branch_num=la_num_list[i], pooling=pooling)
            self.nets.append(net)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        if self.conv_shared:
            x = self.conv(x)

        outputs = []
        for net in self.nets:
            output = net(x).unsqueeze(dim=0)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)

        return outputs

class TaskSpecificSubnet(torch.nn.Module):
    def __init__(self, feature_dim=512, drop_rate=0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Dropout(drop_rate),)

    def forward(self, x):
        return self.feature(x)

class TaskSpecificSubnets(torch.nn.Module):
    def __init__(self, branch_num=11):
        super().__init__()

        self.branch_num = branch_num
        self.nets = nn.ModuleList()
        for i in range(self.branch_num):
            net = TaskSpecificSubnet(drop_rate=0.5)
            self.nets.append(net)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        outputs = []
        for i in range(self.branch_num):
            net = self.nets[i]
            output = net(x[i]).unsqueeze(dim=0)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)

        return outputs

class OutputModule(torch.nn.Module):
    def __init__(self, feature_dim=512, output_type="Dict"):
        super().__init__()
        self.output_sizes = [[2],
                             [1, 2],
                             [7, 2],
                             [2 for j in range(6)],
                             [2 for j in range(10)],
                             [2 for j in range(5)],
                             [2, 2],
                             [2 for j in range(4)],
                             [2 for j in range(6)],
                             [2, 2],
                             [2, 2]]

        self.output_fcs = nn.ModuleList()
        for i in range(0, len(self.output_sizes)):
            for j in range(len(self.output_sizes[i])):
                output_fc = nn.Linear(feature_dim, self.output_sizes[i][j])
                self.output_fcs.append(output_fc)

        self.task_names = [
            'Age', 'Attractive', 'Blurry', 'Chubby', 'Heavy Makeup', 'Gender', 'Oval Face', 'Pale Skin',
            'Smiling', 'Young',
            'Bald', 'Bangs', 'Black Hair', 'Blond Hair', 'Brown Hair', 'Gray Hair', 'Receding Hairline',
            'Straight Hair', 'Wavy Hair', 'Wearing Hat',
            'Arched Eyebrows', 'Bags Under Eyes', 'Bushy Eyebrows', 'Eyeglasses', 'Narrow Eyes', 'Big Nose',
            'Pointy Nose', 'High Cheekbones', 'Rosy Cheeks', 'Wearing Earrings',
            'Sideburns', r"Five O'Clock Shadow", 'Big Lips', 'Mouth Slightly Open', 'Mustache',
            'Wearing Lipstick', 'No Beard', 'Double Chin', 'Goatee', 'Wearing Necklace',
            'Wearing Necktie', 'Expression', 'Recognition']  # Total:43

        self.output_type = output_type

        self.apply(self._init_weights)

    def set_output_type(self, output_type):
        self.output_type = output_type

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, embedding):

        outputs = []

        k = 0
        for i in range(0, len(self.output_sizes)):
            for j in range(len(self.output_sizes[i])):
                output_fc = self.output_fcs[k]
                output = output_fc(x[i])
                outputs.append(output)
                k += 1

        [gender,
         age, young,
         expression, smiling,
         attractive, blurry, chubby, heavy_makeup, oval_face, pale_skin,
         bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline, straight_hair, wavy_hair,
         wearing_hat,
         arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses, narrow_eyes,
         big_nose, pointy_nose,
         high_cheekbones, rosy_cheeks, wearing_earrings, sideburns,
         five_o_clock_shadow, big_lips, mouth_slightly_open, mustache, wearing_lipstick, no_beard,
         double_chin, goatee,
         wearing_necklace, wearing_necktie] = outputs

        outputs = [age, attractive, blurry, chubby, heavy_makeup, gender, oval_face, pale_skin, smiling, young,
                   bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline,
                   straight_hair, wavy_hair, wearing_hat,
                   arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses, narrow_eyes, big_nose,
                   pointy_nose, high_cheekbones, rosy_cheeks, wearing_earrings,
                   sideburns, five_o_clock_shadow, big_lips, mouth_slightly_open, mustache,
                   wearing_lipstick, no_beard, double_chin, goatee, wearing_necklace,
                   wearing_necktie, expression]  # Total:42

        outputs.append(embedding)

        result = dict()
        for j in range(43):
            result[self.task_names[j]] = outputs[j]

        if self.output_type == "Dict":
            return result
        elif self.output_type == "List":
            return outputs
        elif self.output_type == "Attribute":
            return outputs[1: 41]
        else:
            return result[self.output_type]


class ModelBox(torch.nn.Module):

    def __init__(self, backbone=None, fam=None, tss=None, om=None,
                 feature="global", output_type="Dict"):
        super().__init__()
        self.backbone = backbone
        self.fam = fam
        self.tss = tss
        self.om = om
        self.output_type = output_type
        if self.om:
            self.om.set_output_type(self.output_type)

        self.feature = feature

    def set_output_type(self, output_type):
        self.output_type = output_type
        if self.om:
            self.om.set_output_type(self.output_type)


    def forward(self, x):

        local_features, global_features, embedding = self.backbone(x)

        if self.feature == "all":
            x = torch.cat([local_features, global_features], dim=1)
        elif self.feature == "global":
            x = global_features
        elif self.feature == "local":
            x = local_features

        x = self.fam(x)
        x = self.tss(x)

        x = self.om(x, embedding)
        return x