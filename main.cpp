#include <opencv2\opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <fstream>
#include "patchmatch.h"


const double MRT = 0.7;


void extract_features(
	const std::vector<cv::String>& images_path,
	std::vector<cv::Mat>& descriptors,
	std::vector<std::vector<cv::KeyPoint>>& keypoints,
	std::vector<std::vector<cv::Vec3b>>& colors)
{
	std::vector<cv::Mat> images;
	for (const auto& image : images_path) images.push_back(cv::imread(image));
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	for (const auto& image : images)
	{
		std::vector<cv::KeyPoint> kp;
		cv::Mat des;
		std::vector<cv::Vec3b> color;

		sift->detectAndCompute(image, cv::noArray(), kp, des);
		keypoints.push_back(kp);
		descriptors.push_back(des);

		for (const auto& k : kp)
			color.push_back(image.at<cv::Vec3b>(k.pt));
		colors.push_back(color);
	}
}


//匹配特征点（两两）

void match_features(const cv::Mat& query, const cv::Mat& train, std::vector<cv::DMatch>& result)
{
	cv::Ptr<cv::FlannBasedMatcher> flann = cv::FlannBasedMatcher::create();
	std::vector<std::vector<cv::DMatch>> matches;

	flann->knnMatch(query, train, matches, 2);

	// Lowe's SIFT matching ratio test(MRT)
	for (const auto& match : matches)
		if (match[0].distance < MRT * match[1].distance) result.push_back(match[0]);
}

//n个图像进行匹配特征点

void match_all_features(const std::vector<cv::Mat>& descriptors_all, std::vector<std::vector<cv::DMatch>>& matches_all)
{
	for (int i = 1; i < descriptors_all.size(); ++i)
	{
		std::vector<cv::DMatch> match;
		match_features(descriptors_all[i - 1], descriptors_all[i], match);
		matches_all.push_back(match);
	}
}

//利用ransac获得本征矩阵E，使结果尽可能可靠
void find_transform(
	const cv::Mat& K,
	const std::vector<cv::Point2d>& p1,
	const std::vector<cv::Point2d>& p2,
	cv::Mat& R, cv::Mat& T, cv::Mat& mask)
{
	cv::Mat E;
	// 求出本质矩阵
	E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, 1000, mask);
	// 回复旋转矩阵和平移矩阵
	int pass_count = cv::recoverPose(E, p1, p2, K, R, T, mask);
}

/**
 * 寻找图与图之间的对应相机旋转以及相机平移
 */

void get_matched_points(
	const std::vector<cv::KeyPoint>& p1, const std::vector<cv::KeyPoint>& p2,
	const std::vector<cv::DMatch>& matches,
	std::vector<cv::Point2d>& src_pts, std::vector<cv::Point2d>& pts_dst)
{
	for (const auto& match : matches)
	{
		src_pts.push_back(p1[match.queryIdx].pt);
		pts_dst.push_back(p2[match.trainIdx].pt);
	}
}

void get_matched_colors(
	const std::vector<cv::Vec3b>& c1, const std::vector<cv::Vec3b>& c2,
	const std::vector<cv::DMatch>& matches,
	std::vector<cv::Vec3b>& src_colors, std::vector<cv::Vec3b>& dst_colors)
{
	for (const auto& match : matches)
	{
		src_colors.push_back(c1[match.queryIdx]);
		dst_colors.push_back(c2[match.trainIdx]);
	}
}
void convert_to_2xN_mat(const std::vector<cv::Point2d>& p, cv::Mat& ph)
{
	for (int i = 0; i < p.size(); ++i)
		ph.at<double>(0, i) = p[i].x, ph.at<double>(1, i) = p[i].y;
}
//进行三维重建，也就是从两个相机的视图中恢复出三维结构
void reconstruct(
	const cv::Mat& K, const cv::Mat& R1, const cv::Mat& T1,
	const cv::Mat& R2, const cv::Mat& T2,
	const std::vector<cv::Point2d>& p1,
	const std::vector<cv::Point2d>& p2,
	std::vector<cv::Point3d>& structure)
{
	cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
	cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
	R1.copyTo(P1.rowRange(0, 3).colRange(0, 3));
	T1.copyTo(P1.rowRange(0, 3).colRange(3, 4));
	R2.copyTo(P2.rowRange(0, 3).colRange(0, 3));
	T2.copyTo(P2.rowRange(0, 3).colRange(3, 4));
	P1 = K * P1, P2 = K * P2;

	// 将 p1 转换为 2 * N 矩阵
	cv::Mat p1_h(2, static_cast<int>(p1.size()), CV_64F);
	cv::Mat p2_h(2, static_cast<int>(p2.size()), CV_64F);
	convert_to_2xN_mat(p1, p1_h);
	convert_to_2xN_mat(p2, p2_h);
	cv::Mat S;

	// 三角化
	cv::triangulatePoints(P1, P2, p1, p2, S);

	for (int i = 0; i < S.cols; ++i)
	{
		structure.push_back(cv::Point3d(S.at<double>(0, i) / S.at<double>(3, i),
			S.at<double>(1, i) / S.at<double>(3, i),
			S.at<double>(2, i) / S.at<double>(3, i)));
	}
}
template<typename T_>
void maskout_points(std::vector<T_>& p, const cv::Mat& mask)
{
	std::vector<T_> temp;
	for (int64_t i = 0; i < mask.rows; ++i)
		if (mask.at<uchar>(i)) temp.push_back(p[i]);
	p.swap(temp);
}
//修改P1点的颜色，同上↑
void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

//这个函数的主要目的是从匹配的特征点中获取对应的三维点（object points）和二维点（image points）。
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int64_t>& struct_indices,//是一个整数向量的引用，它存储了结构索引。这个向量的每一个元素都对应于一个特征点匹配（feature point match）。元素的值是该特征点在三维结构中的索引。如果该特征点没有对应的三维点，那么索引值就会是-1。
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,//三维点
	vector<Point2f>& image_points)//二维点
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0) continue;                    //<0表示该特征点没有对应的三维点

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);
	}
}
//这个函数的主要目的是将当前结构（structure）和下一个结构（next_structure）融合在一起。
void fusion_structure(
	const std::vector<cv::DMatch>& matches,
	std::vector<int64_t>& struct_indices,
	std::vector<int64_t>& next_struct_indices,
	std::vector<cv::Point3d>& structure,
	const std::vector<cv::Point3d>& next_structure,
	std::vector<cv::Vec3b>& colors,
	const std::vector<cv::Vec3b>& next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int64_t query_idx = matches[i].queryIdx;
		int64_t train_idx = matches[i].trainIdx;
		int64_t struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void init_structure(
	const cv::Mat& K,
	const std::vector<std::vector<cv::KeyPoint>>& keypoints_all,
	const std::vector<std::vector<cv::Vec3b>>& colors_all,
	const std::vector<std::vector<cv::DMatch>>& matches_all,
	std::vector<cv::Point3d>& structure,
	std::vector<cv::Mat>& rotations,
	std::vector<cv::Mat>& motions,
	std::vector<std::vector<int64_t>>& correspond_struct_idx,
	std::vector<cv::Vec3b>& colors)
{
	std::vector<cv::Point2d> p1, p2;
	std::vector<cv::Vec3b> c2;

	get_matched_points(keypoints_all[0], keypoints_all[1], matches_all[0], p1, p2);
	get_matched_colors(colors_all[0], colors_all[1], matches_all[0], colors, c2);

	cv::Mat R, T, mask;

	find_transform(K, p1, p2, R, T, mask);
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_points(colors, mask);

	// 设置第一个相机的变换矩阵，作为剩下摄像机矩阵变换的基准
	cv::Mat R0 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat T0 = cv::Mat::zeros(3, 1, CV_64F);

	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	rotations.push_back(R0);
	rotations.push_back(R);
	motions.push_back(T0);
	motions.push_back(T);
	for (const auto& keypoints : keypoints_all)
		correspond_struct_idx.push_back(std::vector<int64_t>(keypoints.size(), -1));

	int idx = 0;
	for (size_t i = 0; i < matches_all[0].size(); ++i) if (mask.at<uchar>(i))
	{
		correspond_struct_idx[0][matches_all[0][i].queryIdx] = idx;
		correspond_struct_idx[1][matches_all[0][i].trainIdx] = idx;
		++idx;
	}
}


////获取目录下的文件名
//void get_file_names(string dir_name, vector<string>& names)
//{
//	names.clear();
//	tinydir_dir dir;
//	tinydir_open(&dir, dir_name.c_str());
//
//	while (dir.has_next)
//	{
//		tinydir_file file;
//		tinydir_readfile(&dir, &file);
//		if (!file.is_dir)
//		{
//			names.push_back(file.path);
//		}
//		tinydir_next(&dir);
//	}
//	tinydir_close(&dir);
//}
struct ReprojectCost
{
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};




void bundle_adjustment(
	cv::Mat& intrinsic,
	std::vector<cv::Mat>& extrinsics,
	std::vector<std::vector<int64_t>>& correspond_struct_idx,
	std::vector<std::vector<cv::KeyPoint>>& key_points_for_all,
	std::vector<cv::Point3d>& structure
)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy
	// load points
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);	// loss function make bundle adjustment robuster.
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		std::vector<int64_t>& point3d_ids = correspond_struct_idx[img_idx];
		std::vector<cv::KeyPoint>& key_points = key_points_for_all[img_idx];
		if (img_idx == 1)
		{
			std::cout << "stop!!!";
		}
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;
			cv::Point2d observed = key_points[point_idx].pt;
			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),			// Intrinsic
				extrinsics[img_idx].ptr<double>(),	// View Rotation and Translation
				&(structure[point3d_id].x)			// Point in 3D space
			);
		}
	}
	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);
	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}
}
//存储点云位置和颜色信息到ply文件里
void save_ply_file(
	const std::string& ply_save_path,
	std::vector<cv::Point3d>& structure,
	const std::vector<cv::Vec3b>& colors
)
{
	// 手动输出点云ply文件
	std::ofstream ply_file(ply_save_path);
	// ply的头部信息
	ply_file << "ply\n";
	ply_file << "format ascii 1.0\n";
	ply_file << "element vertex " << structure.size() << "\n";
	ply_file << "property float x\n";
	ply_file << "property float y\n";
	ply_file << "property float z\n";
	ply_file << "property uchar red\n";
	ply_file << "property uchar green\n";
	ply_file << "property uchar blue\n";
	ply_file << "end_header\n";
	// 写入点云数据
	for (int i = 0; i < structure.size(); ++i)
	{
		ply_file << structure[i].x << " " << structure[i].y << " " << structure[i].z << " "
			<< static_cast<int>(colors[i][2]) << " "
			<< static_cast<int>(colors[i][1]) << " "
			<< static_cast<int>(colors[i][0]) << std::endl;
	}
	ply_file.close();
}

void main()
{
	std::vector<std::string> images_path;
	//path to photo
	images_path.push_back("B21.jpg");
	images_path.push_back("B22.jpg");
	//images_path.push_back("B23.jpg");
	//images_path.push_back("B24.jpg");
	//images_path.push_back("B25.jpg");
	
	

	//本征矩阵
	Mat K(Matx33d(
		719.5459, 0, 0,    //719.5459
		0, 719.5459, 0,
		0, 0, 1));

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//提取所有图像的特征
	extract_features(images_path, descriptor_for_all, key_points_for_all, colors_for_all);
	//对所有图像进行顺次的特征匹配
	match_all_features(descriptor_for_all, matches_for_all);

	vector<Point3d> structure;
	vector<vector<int64_t>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//初始化结构（三维点云）
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		rotations,
		motions,
		correspond_struct_idx,
		colors
	);

	//增量方式重建剩余的图像
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;
		//Mat mask;

		//获取第i幅图像中匹配点对应的三维点，以及在第i+1幅图像中对应的像素点
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i + 1],
			object_points,
			image_points
		);

		//求解变换矩阵
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		//保存变换矩阵
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2d> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//根据之前求得的R，T进行三维重建
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//将新的重建结果与之前的融合
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1
		);
	}


	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<Mat> extrinsics;
	for (size_t i = 0; i < rotations.size(); ++i)
	{
		Mat extrinsic(6, 1, CV_64FC1);
		Mat r;
		Rodrigues(rotations[i], r);

		r.copyTo(extrinsic.rowRange(0, 3));
		motions[i].copyTo(extrinsic.rowRange(3, 6));

		extrinsics.push_back(extrinsic);
	}


	save_ply_file("./result/output_test_BA.ply", structure, colors);


	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
	//以上为bundlead_justment算法
	for (int i = 0; i < intrinsic.rows; i++) { for (int j = 0; j < intrinsic.cols; j++) { cout << intrinsic.at<double>(i, j) << " "; } cout << endl; }



	//以上为稀疏点云部分

	testImage(images_path, 8);//生成稠密点云（效果还不是很好。。而且很慢。。）
}