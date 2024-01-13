#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h> 
#include <pcl/common/distances.h>
#include <onnxruntime_cxx_api.h>

#include "knn_.h"


const int k_n = 16;
const int num_classes = 19;
const int num_points = 4096 * 11;
const int num_layers = 4;
const float test_smooth = 0.98;


std::vector<std::vector<long>> knn_search(pcl::PointCloud<pcl::PointXYZ>::Ptr& support_pts, pcl::PointCloud<pcl::PointXYZ>::Ptr& query_pts, int k)
{
	float* points = new float[support_pts->size() * 3];
	for (size_t i = 0; i < support_pts->size(); i++)
	{
		points[3 * i + 0] = support_pts->points[i].x;
		points[3 * i + 1] = support_pts->points[i].y;
		points[3 * i + 2] = support_pts->points[i].z;
	}

	float* queries = new float[query_pts->size() * 3];
	for (size_t i = 0; i < query_pts->size(); i++)
	{
		queries[3 * i + 0] = query_pts->points[i].x;
		queries[3 * i + 1] = query_pts->points[i].y;
		queries[3 * i + 2] = query_pts->points[i].z;
	}

	long* indices = new long[query_pts->size() * k];
	cpp_knn_omp(points, support_pts->size(), 3, queries, query_pts->size(), k, indices);

	std::vector<std::vector<long>> neighbour_idx(query_pts->size(), std::vector<long>(k));
	for (size_t i = 0; i < query_pts->size(); i++)
	{
		for (size_t j = 0; j < k; j++)
		{
			neighbour_idx[i][j] = indices[k * i + j];
		}
	}
	return neighbour_idx;
}


int main()
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "randla-net");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	const wchar_t* model_path = L"randla-net.onnx";
	Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;

	std::vector<const char*>  input_node_names;
	for (size_t i = 0; i < session.GetInputCount(); i++)
	{
		input_node_names.push_back(session.GetInputName(i, allocator));
	}

	std::vector<const char*> output_node_names;
	for (size_t i = 0; i < session.GetOutputCount(); i++)
	{
		output_node_names.push_back(session.GetOutputName(i, allocator));
	}

	float x, y, z;
	pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
	std::ifstream infile_points("000000.txt");
	while (infile_points >> x >> y >> z)
	{
		points->push_back(pcl::PointXYZ(x, y, z));
	}

	std::vector<float> possibility(points->size(), 0);
	std::vector<float> min_possibility = { 0 };
	std::vector<std::vector<float>> test_probs(points->size(), std::vector<float>(num_classes, 0));

	pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
	std::ifstream infile_pc("000000.pkl", std::ios::binary);
	while (infile_pc >> x >> y >> z)
	{
		pc->push_back(pcl::PointXYZ(x, y, z));
	}

	std::vector<float> labels(pc->size(), 0);

	while (*std::min_element(min_possibility.begin(), min_possibility.end()) < 0.5)
	{
		int cloud_ind = std::min_element(min_possibility.begin(), min_possibility.end()) - min_possibility.begin();
		int pick_idx = std::min_element(possibility.begin(), possibility.end()) - possibility.begin();

		pcl::PointXYZ center_point = pc->points[pick_idx];

		pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
		kdtree->setInputCloud(pc);

		std::vector<int> selected_idx(num_points);
		std::vector<float> distances(num_points);
		kdtree->nearestKSearch(center_point, num_points, selected_idx, distances);

		pcl::PointCloud<pcl::PointXYZ>::Ptr selected_pc(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*pc, selected_idx, *selected_pc);

		std::vector<float> selected_labels(num_points);
		for (size_t i = 0; i < num_points; i++)
		{
			selected_labels[i] = labels[selected_idx[i]];
		}

		std::vector<float> dists(num_points);
		for (size_t i = 0; i < num_points; i++)
		{
			dists[i] = pcl::squaredEuclideanDistance(selected_pc->points[i], pc->points[pick_idx]);
		}
		float max_dists = *std::max_element(dists.begin(), dists.end());

		std::vector<float> delta(num_points);
		for (size_t i = 0; i < num_points; i++)
		{
			delta[i] = pow(1 - dists[i] / max_dists, 2);
			possibility[selected_idx[i]] += delta[i];
		}
			
		min_possibility[cloud_ind] = *std::min_element(possibility.begin(), possibility.end());

		pcl::PointCloud<pcl::PointXYZ>::Ptr features(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::copyPointCloud(*selected_pc, *features);
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> input_points;
		std::vector<std::vector<std::vector<long>>> input_neighbors, input_pools, input_up_samples;
		for (size_t i = 0; i < num_layers; i++)
		{
			std::vector<std::vector<long>> neighbour_idx = knn_search(selected_pc, selected_pc, k_n);
			pcl::PointCloud<pcl::PointXYZ>::Ptr sub_points(new pcl::PointCloud<pcl::PointXYZ>);
			std::vector<int> index(selected_pc->size() / 4);
			std::iota(index.begin(), index.end(), 0);
			pcl::copyPointCloud(*selected_pc, index, *sub_points);
			std::vector<std::vector<long>> pool_i(selected_pc->size() / 4);
			std::copy(neighbour_idx.begin(), neighbour_idx.begin() + selected_pc->size() / 4, pool_i.begin());
			std::vector<std::vector<long>> up_i = knn_search(sub_points, selected_pc, 1);
			input_points.push_back(selected_pc);
			input_neighbors.push_back(neighbour_idx);
			input_pools.push_back(pool_i);
			input_up_samples.push_back(up_i);
			selected_pc = sub_points;
		}
		const size_t xyz1_size = 1 * input_points[0]->size() * 3;
		std::vector<float> xyz1_values(xyz1_size);
		for (size_t i = 0; i < input_points[0]->size(); i++)
		{
			xyz1_values[3 * i + 0] = input_points[0]->points[i].x;
			xyz1_values[3 * i + 1] = input_points[0]->points[i].y;
			xyz1_values[3 * i + 2] = input_points[0]->points[i].z;
		}
		std::vector<int64_t> xyz1_dims = { 1, (int64_t)input_points[0]->size(), 3 };
		auto xyz1_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value xyz1_tensor = Ort::Value::CreateTensor<float>(xyz1_memory, xyz1_values.data(), xyz1_size, xyz1_dims.data(), xyz1_dims.size());

		const size_t xyz2_size = 1 * input_points[1]->size() * 3;
		std::vector<float> xyz2_values(xyz2_size);
		for (size_t i = 0; i < input_points[1]->size(); i++)
		{
			xyz2_values[3 * i + 0] = input_points[1]->points[i].x;
			xyz2_values[3 * i + 1] = input_points[1]->points[i].y;
			xyz2_values[3 * i + 2] = input_points[1]->points[i].z;
		}
		std::vector<int64_t> xyz2_dims = { 1, (int64_t)input_points[1]->size(), 3 };
		auto xyz2_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value xyz2_tensor = Ort::Value::CreateTensor<float>(xyz2_memory, xyz2_values.data(), xyz2_size, xyz2_dims.data(), xyz2_dims.size());

		const size_t xyz3_size = 1 * input_points[2]->size() * 3;
		std::vector<float> xyz3_values(xyz3_size);
		for (size_t i = 0; i < input_points[2]->size(); i++)
		{
			xyz3_values[3 * i + 0] = input_points[2]->points[i].x;
			xyz3_values[3 * i + 1] = input_points[2]->points[i].y;
			xyz3_values[3 * i + 2] = input_points[2]->points[i].z;
		}
		std::vector<int64_t> xyz3_dims = { 1, (int64_t)input_points[2]->size(), 3 };
		auto xyz3_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value xyz3_tensor = Ort::Value::CreateTensor<float>(xyz3_memory, xyz3_values.data(), xyz3_size, xyz3_dims.data(), xyz3_dims.size());

		const size_t xyz_size = 1 * input_points[3]->size() * 3;
		std::vector<float> xyz_values(xyz_size);
		for (size_t i = 0; i < input_points[3]->size(); i++)
		{
			xyz_values[3 * i + 0] = input_points[3]->points[i].x;
			xyz_values[3 * i + 1] = input_points[3]->points[i].y;
			xyz_values[3 * i + 2] = input_points[3]->points[i].z;
		}
		std::vector<int64_t> xyz_dims = { 1, (int64_t)input_points[3]->size(), 3 };
		auto xyz_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value xyz_tensor = Ort::Value::CreateTensor<float>(xyz_memory, xyz_values.data(), xyz_size, xyz_dims.data(), xyz_dims.size());

		const size_t neigh_idx1_size = 1 * input_neighbors[0].size() * 16;
		std::vector<int64_t> neigh_idx1_values(neigh_idx1_size);
		for (size_t i = 0; i < input_neighbors[0].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				neigh_idx1_values[16 * i + j] = input_neighbors[0][i][j];
			}
		}
		std::vector<int64_t> neigh_idx1_dims = { 1, (int64_t)input_neighbors[0].size(), 16 };
		auto neigh_idx1_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value neigh_idx1_tensor = Ort::Value::CreateTensor<int64_t>(neigh_idx1_memory, neigh_idx1_values.data(), neigh_idx1_size, neigh_idx1_dims.data(), neigh_idx1_dims.size());

		const size_t neigh_idx2_size = 1 * input_neighbors[1].size() * 16;
		std::vector<int64_t> neigh_idx2_values(neigh_idx2_size);
		for (size_t i = 0; i < input_neighbors[1].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				neigh_idx2_values[16 * i + j] = input_neighbors[1][i][j];
			}
		}
		std::vector<int64_t> neigh_idx2_dims = { 1, (int64_t)input_neighbors[1].size(), 16 };
		auto neigh_idx2_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value neigh_idx2_tensor = Ort::Value::CreateTensor<int64_t>(neigh_idx2_memory, neigh_idx2_values.data(), neigh_idx2_size, neigh_idx2_dims.data(), neigh_idx2_dims.size());

		const size_t neigh_idx3_size = 1 * input_neighbors[2].size() * 16;
		std::vector<int64_t> neigh_idx3_values(neigh_idx3_size);
		for (size_t i = 0; i < input_neighbors[2].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				neigh_idx3_values[16 * i + j] = input_neighbors[2][i][j];
			}
		}
		std::vector<int64_t> neigh_idx3_dims = { 1, (int64_t)input_neighbors[2].size(), 16 };
		auto neigh_idx3_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value neigh_idx3_tensor = Ort::Value::CreateTensor<int64_t>(neigh_idx3_memory, neigh_idx3_values.data(), neigh_idx3_size, neigh_idx3_dims.data(), neigh_idx3_dims.size());

		const size_t neigh_idx_size = 1 * input_neighbors[3].size() * 16;
		std::vector<int64_t> neigh_idx_values(neigh_idx_size);
		for (size_t i = 0; i < input_neighbors[3].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				neigh_idx_values[16 * i + j] = input_neighbors[3][i][j];
			}
		}
		std::vector<int64_t> neigh_idx_dims = { 1, (int64_t)input_neighbors[3].size(), 16 };
		auto neigh_idx_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value neigh_idx_tensor = Ort::Value::CreateTensor<int64_t>(neigh_idx_memory, neigh_idx_values.data(), neigh_idx_size, neigh_idx_dims.data(), neigh_idx_dims.size());
	
		const size_t sub_idx8_size = 1 * input_pools[0].size() * 16;
		std::vector<int64_t> sub_idx8_values(sub_idx8_size);
		for (size_t i = 0; i < input_pools[0].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				sub_idx8_values[16 * i + j] = input_pools[0][i][j];
			}
		}
		std::vector<int64_t> sub_idx8_dims = { 1, (int64_t)input_pools[0].size(), 16 };
		auto sub_idx8_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value sub_idx8_tensor = Ort::Value::CreateTensor<int64_t>(sub_idx8_memory, sub_idx8_values.data(), sub_idx8_size, sub_idx8_dims.data(), sub_idx8_dims.size());

		const size_t sub_idx9_size = 1 * input_pools[1].size() * 16;
		std::vector<int64_t> sub_idx9_values(sub_idx9_size);
		for (size_t i = 0; i < input_pools[1].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				sub_idx9_values[16 * i + j] = input_pools[1][i][j];
			}
		}
		std::vector<int64_t> sub_idx9_dims = { 1, (int64_t)input_pools[1].size(), 16 };
		auto sub_idx9_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value sub_idx9_tensor = Ort::Value::CreateTensor<int64_t>(sub_idx9_memory, sub_idx9_values.data(), sub_idx9_size, sub_idx9_dims.data(), sub_idx9_dims.size());

		const size_t sub_idx10_size = 1 * input_pools[2].size() * 16;
		std::vector<int64_t> sub_idx10_values(sub_idx10_size);
		for (size_t i = 0; i < input_pools[2].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				sub_idx10_values[16 * i + j] = input_pools[2][i][j];
			}
		}
		std::vector<int64_t> sub_idx10_dims = { 1, (int64_t)input_pools[2].size(), 16 };
		auto sub_idx10_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value sub_idx10_tensor = Ort::Value::CreateTensor<int64_t>(sub_idx10_memory, sub_idx10_values.data(), sub_idx10_size, sub_idx10_dims.data(), sub_idx10_dims.size());

		const size_t sub_idx11_size = 1 * input_pools[3].size() * 16;
		std::vector<int64_t> sub_idx11_values(sub_idx11_size);
		for (size_t i = 0; i < input_pools[3].size(); i++)
		{
			for (size_t j = 0; j < 16; j++)
			{
				sub_idx11_values[16 * i + j] = input_pools[3][i][j];
			}
		}
		std::vector<int64_t> sub_idx11_dims = { 1, (int64_t)input_pools[3].size(), 16 };
		auto sub_idx11_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value sub_idx11_tensor = Ort::Value::CreateTensor<int64_t>(sub_idx11_memory, sub_idx11_values.data(), sub_idx11_size, sub_idx11_dims.data(), sub_idx11_dims.size());
	
		const size_t interp_idx12_size = 1 * input_up_samples[0].size() * 1;
		std::vector<int64_t> interp_idx12_values(interp_idx12_size);
		for (size_t i = 0; i < input_up_samples[0].size(); i++)
		{
			interp_idx12_values[i] = input_up_samples[0][i][0];
		}
		std::vector<int64_t> interp_idx12_dims = { 1, (int64_t)input_up_samples[0].size(), 1 };
		auto interp_idx12_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value interp_idx12_tensor = Ort::Value::CreateTensor<int64_t>(interp_idx12_memory, interp_idx12_values.data(), interp_idx12_size, interp_idx12_dims.data(), interp_idx12_dims.size());

		const size_t interp_idx13_size = 1 * input_up_samples[1].size() * 1;
		std::vector<int64_t> interp_idx13_values(interp_idx13_size);
		for (size_t i = 0; i < input_up_samples[1].size(); i++)
		{
			interp_idx13_values[i] = input_up_samples[1][i][0];
		}
		std::vector<int64_t> interp_idx13_dims = { 1, (int64_t)input_up_samples[1].size(), 1 };
		auto interp_idx13_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value interp_idx13_tensor = Ort::Value::CreateTensor<int64_t>(interp_idx13_memory, interp_idx13_values.data(), interp_idx13_size, interp_idx13_dims.data(), interp_idx13_dims.size());

		const size_t interp_idx14_size = 1 * input_up_samples[2].size() * 1;
		std::vector<int64_t> interp_idx14_values(interp_idx14_size);
		for (size_t i = 0; i < input_up_samples[2].size(); i++)
		{
			interp_idx14_values[i] = input_up_samples[2][i][0];
		}
		std::vector<int64_t> interp_idx14_dims = { 1, (int64_t)input_up_samples[2].size(), 1 };
		auto interp_idx14_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value interp_idx14_tensor = Ort::Value::CreateTensor<int64_t>(interp_idx14_memory, interp_idx14_values.data(), interp_idx14_size, interp_idx14_dims.data(), interp_idx14_dims.size());

		const size_t interp_idx15_size = 1 * input_up_samples[3].size() * 1;
		std::vector<int64_t> interp_idx15_values(interp_idx15_size);
		for (size_t i = 0; i < input_up_samples[3].size(); i++)
		{
			interp_idx15_values[i] = input_up_samples[3][i][0];
		}
		std::vector<int64_t> interp_idx15_dims = { 1, (int64_t)input_up_samples[3].size(), 1 };
		auto interp_idx15_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value interp_idx15_tensor = Ort::Value::CreateTensor<int64_t>(interp_idx15_memory, interp_idx15_values.data(), interp_idx15_size, interp_idx15_dims.data(), interp_idx15_dims.size());

		const size_t features_size = 1 * 3 * features->size();
		std::vector<float> features_values(features_size);
		for (size_t i = 0; i < features->size(); i++)
		{
			features_values[features->size() * 0 + i] = features->points[i].x;
			features_values[features->size() * 1 + i] = features->points[i].y;
			features_values[features->size() * 2 + i] = features->points[i].z;
		}
		std::vector<int64_t> features_dims = { 1, 3, (int64_t)features->size() };
		auto features_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value features_tensor = Ort::Value::CreateTensor<float>(features_memory, features_values.data(), features_size, features_dims.data(), features_dims.size());

		const size_t labels_size = 1 * selected_labels.size();
		std::vector<int64_t> labels_values(labels_size);
		for (size_t i = 0; i < selected_labels.size(); i++)
		{
			labels_values[i] = selected_labels[i];
		}
		std::vector<int64_t> labels_dims = { 1, (int64_t)selected_labels.size() };
		auto labels_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value labels_tensor = Ort::Value::CreateTensor<int64_t>(labels_memory, labels_values.data(), labels_size, labels_dims.data(), labels_dims.size());

		std::vector<Ort::Value> inputs;
		inputs.push_back(std::move(xyz1_tensor));
		inputs.push_back(std::move(xyz2_tensor));
		inputs.push_back(std::move(xyz3_tensor));
		inputs.push_back(std::move(xyz_tensor));
		inputs.push_back(std::move(neigh_idx1_tensor));
		inputs.push_back(std::move(neigh_idx2_tensor));
		inputs.push_back(std::move(neigh_idx3_tensor));
		inputs.push_back(std::move(neigh_idx_tensor));
		inputs.push_back(std::move(sub_idx8_tensor));
		inputs.push_back(std::move(sub_idx9_tensor));
		inputs.push_back(std::move(sub_idx10_tensor));
		inputs.push_back(std::move(sub_idx11_tensor));
		inputs.push_back(std::move(interp_idx12_tensor));
		inputs.push_back(std::move(interp_idx13_tensor));
		inputs.push_back(std::move(interp_idx14_tensor));
		inputs.push_back(std::move(interp_idx15_tensor));
		inputs.push_back(std::move(features_tensor));
		inputs.push_back(std::move(labels_tensor));

		std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

		const float* output = outputs[18].GetTensorData<float>();
		std::vector<int64_t> output_dims = outputs[18].GetTensorTypeAndShapeInfo().GetShape(); //1*19*45056
		size_t count = outputs[18].GetTensorTypeAndShapeInfo().GetElementCount();
		std::vector<float> pred(output, output + count);
		std::vector<std::vector<float>> probs(output_dims[2], std::vector<float>(output_dims[1])); //45056*19
		for (size_t i = 0; i < output_dims[2]; i++)
		{
			for (size_t j = 0; j < output_dims[1]; j++)
			{
				probs[i][j] = pred[j * output_dims[2] + i];
			}
		}

		std::vector<int> inds = selected_idx;
		int c_i = cloud_ind;
		for (size_t i = 0; i < inds.size(); i++)
		{
			for (size_t j = 0; j < num_classes; j++)
			{
				test_probs[inds[i]][j] = test_smooth * test_probs[inds[i]][j] + (1 - test_smooth) * probs[i][j];
			}
		}
	}

	std::vector<int> pred(test_probs.size());
	std::fstream output("output.txt", 'w');
	for (size_t i = 0; i < test_probs.size(); i++)
	{
		pred[i] = max_element(test_probs[i].begin(), test_probs[i].end()) - test_probs[i].begin() + 1;
		output << points->points[i].x << " " << points->points[i].y << " " << points->points[i].z << " " << pred[i] << std::endl;
	}

	return 0;
}