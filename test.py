import pickle
import numpy as np
import torch
from network.RandLANet import Network
from utils.data_process import DataProcessing as DP
from utils.config import ConfigSemanticKITTI as cfg


np.random.seed(0)
k_n = 16 
num_points = 4096 * 11 
num_layers = 4
num_classes = 19
sub_sampling_ratio = [4, 4, 4, 4]  


if __name__ == '__main__':
    net = Network(cfg).to(torch.device("cpu"))
    checkpoint = torch.load("pretrain_model/checkpoint.tar", map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
    points = np.load('./data/08/velodyne/000000.npy') 
    possibility = np.zeros(points.shape[0]) * 1e-3 #[np.random.rand(points.shape[0]) * 1e-3]
    min_possibility = [float(np.min(possibility[-1]))]
    probs = [np.zeros(shape=[points.shape[0], num_classes], dtype=np.float32)]
    test_probs = probs
    test_smooth = 0.98
    
    import onnxruntime     
    onnx_session = onnxruntime.InferenceSession("randla-net.onnx", providers=['CPUExecutionProvider'])

    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)

    output_name = []
    for node in onnx_session.get_outputs():
        output_name.append(node.name)
        
    net.eval()
    with torch.no_grad():
        with open('./data/08/KDTree/000000.pkl', 'rb') as f:
            tree = pickle.load(f)
            pc = np.array(tree.data, copy=False)
            labels = np.zeros(np.shape(pc)[0])    
        while np.min(min_possibility) <= 0.5:
            cloud_ind = int(np.argmin(min_possibility))
            pick_idx = np.argmin(possibility)        
            center_point = pc[pick_idx, :].reshape(1, -1)
            selected_idx = tree.query(center_point, num_points)[1][0]
            selected_pc = pc[selected_idx]
            selected_labels = labels[selected_idx]   
            dists = np.sum(np.square((selected_pc - pc[pick_idx])), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            possibility[selected_idx] += delta  # possibility[193] += delta[1], possibility[20283] += delta[45055]
            min_possibility[cloud_ind] = np.min(possibility)
            
            batch_pc = np.expand_dims(selected_pc, 0)
            batch_label = np.expand_dims(selected_labels, 0)
            batch_pc_idx = np.expand_dims(selected_idx, 0)
            batch_cloud_idx = np.expand_dims(np.array([cloud_ind], dtype=np.int32), 0)
            features = batch_pc
            input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
            for i in range(num_layers):
                neighbour_idx = DP.knn_search(batch_pc, batch_pc, k_n)
                sub_points = batch_pc[:, :batch_pc.shape[1] // sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :batch_pc.shape[1] // sub_sampling_ratio[i], :]   
                up_i = DP.knn_search(sub_points, batch_pc, 1)
                input_points.append(batch_pc)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_pc = sub_points
            flat_inputs = input_points + input_neighbors + input_pools + input_up_samples
            flat_inputs += [features, batch_label, batch_pc_idx, batch_cloud_idx]

            batch_data, inputs = {}, {}    
                  
            batch_data['xyz'] = []
            for tmp in flat_inputs[:num_layers]:
                batch_data['xyz'].append(torch.from_numpy(tmp).float())
            inputs['xyz.1'] = flat_inputs[:num_layers][0].astype(np.float32)
            inputs['xyz.2'] = flat_inputs[:num_layers][1].astype(np.float32)
            inputs['xyz.3'] = flat_inputs[:num_layers][2].astype(np.float32)
            inputs['xyz'] = flat_inputs[:num_layers][3].astype(np.float32)
            
            batch_data['neigh_idx'] = []
            for tmp in flat_inputs[num_layers: 2 * num_layers]:
                batch_data['neigh_idx'].append(torch.from_numpy(tmp).long())
            inputs['neigh_idx.1'] = flat_inputs[num_layers: 2 * num_layers][0].astype(np.int64)
            inputs['neigh_idx.2'] = flat_inputs[num_layers: 2 * num_layers][1].astype(np.int64)
            inputs['neigh_idx.3'] = flat_inputs[num_layers: 2 * num_layers][2].astype(np.int64)
            inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers][3].astype(np.int64)
            
            batch_data['sub_idx'] = []
            for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
                batch_data['sub_idx'].append(torch.from_numpy(tmp).long())
            inputs['8'] = flat_inputs[2 * num_layers:3 * num_layers][0].astype(np.int64)
            inputs['9'] = flat_inputs[2 * num_layers:3 * num_layers][1].astype(np.int64)
            inputs['10'] = flat_inputs[2 * num_layers:3 * num_layers][2].astype(np.int64)
            inputs['11'] = flat_inputs[2 * num_layers:3 * num_layers][3].astype(np.int64)
            
            batch_data['interp_idx'] = []
            for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
                batch_data['interp_idx'].append(torch.from_numpy(tmp).long())
            inputs['12'] = flat_inputs[3 * num_layers:4 * num_layers][0].astype(np.int64)
            inputs['13'] = flat_inputs[3 * num_layers:4 * num_layers][1].astype(np.int64)
            inputs['14'] = flat_inputs[3 * num_layers:4 * num_layers][2].astype(np.int64)
            inputs['15'] = flat_inputs[3 * num_layers:4 * num_layers][3].astype(np.int64)
            
            batch_data['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
            inputs['input.1'] = np.swapaxes(flat_inputs[4 * num_layers], 1, 2).astype(np.float32)
            
            batch_data['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
            inputs['17'] = flat_inputs[4 * num_layers + 1].astype(np.int64)
            
            input_inds = flat_inputs[4 * num_layers + 2]
            cloud_inds = flat_inputs[4 * num_layers + 3]
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(num_layers):
                        batch_data[key][i] = batch_data[key][i]
                else:
                    batch_data[key] = batch_data[key]
                    
            end_points = net(batch_data)
            outputs = onnx_session.run(None, inputs)      
            
            end_points['logits'] = end_points['logits'].transpose(1, 2).cpu().numpy()
            for j in range(end_points['logits'].shape[0]):
                probs = end_points['logits'][j]
                inds = input_inds[j]
                c_i = cloud_inds[j][0]
                test_probs[c_i][inds] = test_smooth * test_probs[c_i][inds] + (1 - test_smooth) * probs #19  (45056, 19)

    for j in range(len(test_probs)): 
        pred = np.argmax(test_probs[j], 1).astype(np.uint32) + 1
        output = np.concatenate((points, pred.reshape(-1, 1)), axis=1)
        np.savetxt('./result/output.txt', output)
