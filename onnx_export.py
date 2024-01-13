import torch
from utils.config import ConfigSemanticKITTI as cfg
from network.RandLANet import Network


model = Network(cfg)
checkpoint = torch.load("./pretrain_model/checkpoint.tar")
model.load_state_dict(checkpoint['model_state_dict'])

input = {}
input['xyz'] = [torch.zeros([1, 45056, 3]), torch.zeros([1, 11264, 3]), torch.zeros([1, 2816, 3]), torch.zeros([1, 704, 3])]
input['neigh_idx'] = [torch.zeros([1, 45056, 16], dtype=torch.int64), torch.zeros([1, 11264, 16], dtype=torch.int64), 
                      torch.zeros([1, 2816, 16], dtype=torch.int64), torch.zeros([1, 704, 16], dtype=torch.int64)]
input['sub_idx'] = [torch.zeros([1, 11264, 16], dtype=torch.int64), torch.zeros([1, 2816, 16], dtype=torch.int64), 
                   torch.zeros([1, 704, 16], dtype=torch.int64), torch.zeros([1, 176, 16], dtype=torch.int64)]
input['interp_idx'] = [torch.zeros([1, 45056, 1], dtype=torch.int64), torch.zeros([1, 11264, 1], dtype=torch.int64), 
                       torch.zeros([1, 2816, 1], dtype=torch.int64), torch.zeros([1, 704, 1], dtype=torch.int64)]
input['features'] = torch.zeros([1, 3, 45056])
input['labels'] = torch.zeros([1, 45056], dtype=torch.int64)
input['logits'] = torch.zeros([1, 19, 45056])

torch.onnx.export(model, input, "randla-net.onnx", opset_version=13)