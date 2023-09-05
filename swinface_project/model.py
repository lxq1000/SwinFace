
from backbones import get_model
from analysis import subnets

def build_model(cfg):

    backbone = get_model(cfg.network, num_features=cfg.embedding_size)

    fam = subnets.FeatureAttentionModule(
        in_chans=cfg.fam_in_chans, kernel_size=cfg.fam_kernel_size, 
        conv_shared=cfg.fam_conv_shared, conv_mode=cfg.fam_conv_mode, 
        channel_attention=cfg.fam_channel_attention, spatial_attention=cfg.fam_spatial_attention,
        pooling=cfg.fam_pooling, la_num_list=cfg.fam_la_num_list)
    tss = subnets.TaskSpecificSubnets()
    om = subnets.OutputModule()

    model = subnets.ModelBox(backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature)

    return model