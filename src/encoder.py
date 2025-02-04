import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


def replace_padding(model):
    """Replace all ZeroPad2d with ReflectionPad2d in the model"""
    modules = list(model.named_children())

    for name, module in modules:
        if len(list(module.children())) > 0:
            replace_padding(module)

        if isinstance(module, nn.Conv2d):
            padding = module.padding
            if padding[0] != 0:
                refl_pad = nn.ReflectionPad2d(padding[0])
                conv = nn.Conv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size[0],
                    module.stride[0],
                    padding=0,
                    bias=(module.bias is not None)
                )

                # Copy the weights and bias
                conv.weight.data = module.weight.data
                if module.bias is not None:
                    conv.bias.data = module.bias.data

                # Get the index of the current module
                for i, (curr_name, curr_module) in enumerate(model._modules.items()):
                    if curr_module == module:
                        # Create new OrderedDict with updated structure
                        from collections import OrderedDict
                        new_modules = OrderedDict()

                        # Copy modules up to this point
                        for j, (n, m) in enumerate(model._modules.items()):
                            if j == i:
                                # Insert our new modules
                                new_name = f"{name}_pad"
                                new_modules[new_name] = refl_pad
                                new_modules[name] = conv
                            else:
                                new_modules[n] = m

                        # Replace the modules in the model
                        model._modules = new_modules
                        break

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        replace_padding(vgg)

        # slices are adjusted because of the reflection padding added
        self.slice1 = vgg[: 3]
        self.slice2 = vgg[3: 10]
        self.slice3 = vgg[10:17]
        self.slice4 = vgg[17: 30]

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, image, output_style_features=False):
        s1 = self.slice1(image)
        s2 = self.slice2(s1)
        s3 = self.slice3(s2)
        s4 = self.slice4(s3)

        if output_style_features:
            return s1, s2, s3, s4
        else:
            return s4
