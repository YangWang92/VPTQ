# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch.nn as nn


# find specific layers in a model
def find_layers(module, target_layers=[nn.Linear], name=''):
    if type(module) in target_layers:
        return {name: module}
    res = {}
    for old_name, child in module.named_children():
        res.update(find_layers(child, target_layers=target_layers, name=name + '.' + old_name if name != '' else old_name))
    return res


def replace_layer(module, target_name, layer, module_name=None):
    for child_name, child_module in module.named_children():
        current_name = child_name if module_name is None else f'{module_name}.{child_name}'
        if target_name == current_name:
            setattr(module, child_name, layer)
            return True 
        else:
            if replace_layer(child_module, target_name, layer, current_name):
                return True 
    return False 
