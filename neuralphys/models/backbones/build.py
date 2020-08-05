def build_backbone(backbone_name, feature_dim):
    if backbone_name == 'hourglass':
        from neuralphys.models.backbones.hg_gn import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=3)
    elif backbone_name == 'hourglass_bn':
        from neuralphys.models.backbones.hg_bn import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=3)
    elif backbone_name == 'hourglass_S':
        from neuralphys.models.backbones.hg_gn_small import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=3)
    elif backbone_name == 'hourglass_S_bn':
        from neuralphys.models.backbones.hg_bn_small import hg
        backbone = hg(depth=3, num_stacks=1, num_blocks=1, num_classes=feature_dim, num_input_c=3)
    else:
        raise NotImplementedError
    return backbone
