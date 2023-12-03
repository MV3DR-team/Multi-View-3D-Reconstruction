#include "nonfree.hpp"
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
#include "tinydir.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "patchmatch.h"
#include <stdio.h>
#include <fstream>
/*图片请用png格式~~~*/
using namespace cv;
using namespace std;


//特征提取函数
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,   //Mat 类是一个用于保存图像数据或者矩阵数据的数据结构，可以说是一个矩阵类
	vector<vector<Vec3b>>& colors_for_all  //uchar类型的，长度为3的vector向量(RGB)
)
{
	key_points_for_all.clear();     //vector.clear():清除向量内的元素
	descriptor_for_all.clear();
	Mat image;
	//最后测得参数0 17 0.0000009 16，但是运行时间较长（巨长）现在这个参数运行较快，
	// 不同得参数组合可能会导致报错！注意！↓
	//读取图像，获取图像特征点，并保存
	Ptr<Feature2D> sift = SIFT::create(0, 3, 0.01, 10);//创建一个sift对象，全称是Scale-Invariant Feature Transform，是与缩放无关的特征检测，
	                                                         //也就是具有尺度不变性。SIFT算法可以说是传统CV领域中达到巅峰的一个算法，这个算法强大到即使对图片进行放缩、变形、模糊、明暗变化、光照变化、添加噪声，
	                                                         // 甚至是使用不同的相机拍摄不同角度的照片的情况下，SIFT都能检测到稳定的特征点，并建立对应关系。
	for (auto it = image_names.begin(); it != image_names.end(); ++it) //vector(数组）
	{
		image = imread(*it);
		if (image.empty()) continue;

		cout << "Extracing features: " << *it << endl;      //每当这行代码被执行时，它都会在新行上打印一条消息，显示正在提取哪个图像的特征。这对于跟踪程序的进度非常有用。

		vector<KeyPoint> key_points;
		Mat descriptor;//图像特征点的标识符
		//偶尔出现内存分配失败的错误
		sift->detectAndCompute(image, noArray(), key_points, descriptor);//检测关键点并计算描述符

		//特征点过少，则排除该图像
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;   //.pt是坐标，创建一个引用 p，指向第 i 个关键点的坐标。Point2f表示一个二维点（浮点
			colors[i] = image.at<Vec3b>(p.y, p.x);  //image.at<Vec3b>(p.y, p.x) 是一个函数调用，它返回图像在 (p.x, p.y) 坐标处的像素值。这个像素值是一个 Vec3b 类型的对象，表示一个 BGR 颜色
		}
		colors_for_all.push_back(colors);//压进去
	}
}
//匹配特征点（两两）
void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{                       //查询      //训练
	vector<vector<DMatch>> knn_matches;//这是一个向量，用于存储匹配的特征点对。
	BFMatcher matcher(NORM_L2);        //这行代码创建了一个暴力匹配器对象，使用 L2 范数进行匹配。
	matcher.knnMatch(query, train, knn_matches, 2);//这行代码找到每个查询描述符的最近邻描述符。

	//获取满足Ratio Test的最小匹配的距离
	//这里使用了Ratio Test方法，即使用KNN算法寻找与该特征最匹配的2个特征，若第一个特征的匹配距离与第二个特征的匹配距离之比小于某一阈值，就接受该匹配，否则视为误匹配
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 2.5 * knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}
	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.76 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 8 * max(min_dist, 10.0f)
			)
			continue;

		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}
//n个图像进行匹配特征点
void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n个图像，两两顺次有 n-1 对匹配
	// 1与2匹配，2与3匹配，3与4匹配，以此类推
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}
//利用ransac获得本征矩阵E，使结果尽可能可靠
bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)//它是一个二值图像，用于表示匹配点是否为内点，你可以把mask看作是一个标记数组，用于标记哪些匹配点是有效的（即内点），哪些是无效的（即外点）。
{
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.2, 1.0, mask);
	                           //相机的焦距（像素）//光点（像素）      //阈值，通过的概率
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;//内点数量 -in- 总匹配点数量
	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//分解本征矩阵，获取相对变换
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
	//这个函数返回一个整数，表示同时位于两个相机前方的点的数量。如果你没有指定相机的内参矩阵，那么这个函数会使用默认值
	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.4)
		return false;

	return true;
}
//从匹配结果中提取出匹配的关键点
void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,//DMatch是OpenCV库中的一个类，它用于存储两个特征点之间的匹配关系。每一个DMatch对象包含以下信息：queryIdx：查询图像中特征点的索引。trainIdx：训练图像中特征点的索引。imgIdx：如果有多个训练图像，这个参数表示特征点所在的训练图像的索引。distance：两个匹配特征点之间的距离。这个距离通常是根据特征描述符（feature descriptors）计算得出的。
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2//这两个参数是输出参数，用于存储匹配的关键点。
)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);//查询
		out_p2.push_back(p2[matches[i].trainIdx].pt);//训练
	}
}
//从匹配结果中提取出匹配的关键点的颜色
void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}
//进行三维重建，也就是从两个相机的视图中恢复出三维结构
void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)//structure:重建后的三维结构
{                                                                                                    //Point3d是模板类，用来表示由x,y,z表示的3D点
	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);//-》改变通道深度？
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);
	//投影后：
	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	//↑调用triangulatePoints函数进行三角测量，得到齐次坐标（三角重建）
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);//s:输出的四维点

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(Point3d(col(0), col(1), col(2)));
	}
}
//修改P1内的点，只有当掩码>0时才会保留
void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
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
// 这个函数的主要目的是将旋转矩阵、运动矩阵、三维结构和颜色信息保存到文件中
void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3d>& structure, vector<Vec3b>& colors)
{                                               //旋转矩阵            //运动矩阵
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();//释放文件对象
}
//这个函数的主要目的是从匹配的特征点中获取对应的三维点（object points）和二维点（image points）。
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,//是一个整数向量的引用，它存储了结构索引。这个向量的每一个元素都对应于一个特征点匹配（feature point match）。元素的值是该特征点在三维结构中的索引。如果该特征点没有对应的三维点，那么索引值就会是-1。
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
	vector<DMatch>& matches,
	vector<int>& struct_indices,    //它存储了结构索引。这个向量的每一个元素都对应于一个特征点匹配（feature point match）,使用引用（&）是为了让函数能够直接操作原始数据
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;      //查询索引：这是指在查询图像中的特征点的索引。所谓的“查询图像”，就是我们想要在其中查找匹配特征的图像。
		                                          //训练索引：这是指在训练图像中的特征点的索引。所谓的“训练图像”，就是我们用来和查询图像进行比较的图像。
                                                 //当我们进行特征匹配时，会从查询图像中选取一个特征点，然后在训练图像中找到与之最匹配的特征点。这两个特征点就形成了一个匹配对，它们各自在自己所在的图像中的位置就是查询索引和训练索引。希望这个解释对你有所帮助！
		int train_idx = matches[i].trainIdx;                                             

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0)                     //若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//若该点在空间中已经存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;//扩充原有的结构
	}
}
//初始化三维结构并且进行基础矩阵的运算
void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,//用于存储结构索引。具体来说，它的每个元素 correspond_struct_idx[i][j] 表示第 i 幅图像中的第 j 个关键点对应的结构索引，若没有对应点，则为-1
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	//计算头两幅图像之间的变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//旋转矩阵和平移向量
	Mat mask;	//mask中大于零的点代表匹配点，等于零代表失配点
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
	find_transform(K, p1, p2, R, T, mask);

	//对头两幅图像进行三维重建
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);//创建一个3X3单位矩阵，元素类型为浮点数
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);//创建一个3X1零矩阵，二者构成了第一幅图的相机位置
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	//保存变换矩阵
	rotations = { R0, R };
	motions = { T0, T };

	//将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//调整结构索引的大小，并将新元素初始化为-1
	}

	//填写头两幅图像的结构索引
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;//处理特征点的配对
	}
}
//获取目录下的文件名
void get_file_names(string dir_name, vector<string>& names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}
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

////进行束调整（bundle adjustment)优化sfm
//void bundle_adjustment(
//	Mat& intrinsic,
//	vector<Mat>& extrinsics,
//	vector<vector<int>>& correspond_struct_idx,//存储结构索引，指向匹配的特征点
//	vector<vector<KeyPoint>>& key_points_for_all,
//	vector<Point3d>& structure
//)
//{
//	ceres::Problem problem;//用于表示一个优化问题，可以使用这个对象来定义和解决一个优化问题
//
//	// load extrinsics (rotations and motions)
//	for (size_t i = 0; i < extrinsics.size(); ++i)
//	{
//		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
//	}
//	// fix the first camera.
//	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());
//
//	// load intrinsic
//	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy，固定第一台相机
//
//	// load points
//	                                     //       分配内存↓
//	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);	// loss function make bundle adjustment robuster.
//	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
//	{
//		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
//		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
//		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
//		{
//			int point3d_id = point3d_ids[point_idx];
//			if (point3d_id < 0)
//				continue;
//
//			Point2d observed = key_points[point_idx].pt;
//			// 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
//			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
//			//创建一个新的自动微分代价函数对象↑
//			//添加一个残差块👇
//			problem.AddResidualBlock(
//				cost_function,
//				loss_function,
//				intrinsic.ptr<double>(),			// 相机的内参
//				extrinsics[img_idx].ptr<double>(),	// 相机的外参
//				&(structure[point3d_id].x)			// 三维空间中的点
//			);
//		}
//	}
//
//	// Solve BA
//	//ceres求解器
//	ceres::Solver::Options ceres_config_options;
//	ceres_config_options.minimizer_progress_to_stdout = false;//优化器的进度不会打印到标准输出
//	ceres_config_options.logging_type = ceres::SILENT;//不打印日志
//	ceres_config_options.num_threads = 1;//ceres求解器在评估雅可比矩阵时只使用一个线程
//	ceres_config_options.preconditioner_type = ceres::JACOBI;
//	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//使用稀疏舒尔线性求解器
//	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;//使用Eigen库中的稀疏线性代数库
//
//	ceres::Solver::Summary summary;
//	ceres::Solve(ceres_config_options, &problem, &summary);//执行优化，结果储存在summary中
//
//
//	//判断结果是否成功
//	if (!summary.IsSolutionUsable())
//	{
//		std::cout << "Bundle Adjustment failed." << std::endl;
//	}
//	else
//	{
//		// Display statistics about the minimization
//		std::cout << std::endl
//			<< "Bundle Adjustment statistics (approximated RMSE):\n"
//			<< " #views: " << extrinsics.size() << "\n"
//			<< " #residuals: " << summary.num_residuals << "\n"
//			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
//			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
//			<< " Time (s): " << summary.total_time_in_seconds << "\n"
//			<< std::endl;
//	}
//}


void bundle_adjustment(
	cv::Mat& intrinsic,
	std::vector<cv::Mat>& extrinsics,
	std::vector<std::vector<int>>& correspond_struct_idx,
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
		std::vector<int>& point3d_ids = correspond_struct_idx[img_idx];
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
			std::cout << img_idx << ":" << point_idx << std::endl;
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
	//ply_file.close();
}

void main()
{
	vector<string> img_names;
	get_file_names("images", img_names);
	
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
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//对所有图像进行顺次的特征匹配
	match_features(descriptor_for_all, matches_for_all);

	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
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
		correspond_struct_idx,
		colors,
		rotations,
		motions
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

		vector<Point2f> p1, p2;
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
	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);
	//以上为bundlead_justment算法
	for (int i = 0; i < intrinsic.rows; i++) { for (int j = 0; j < intrinsic.cols; j++) { cout << intrinsic.at<double>(i, j) << " "; } cout << endl; }
	

	save_ply_file("output_test_BA.ply", structure, colors);
	save_structure(".\\Viewer\\structure.yml", rotations, motions, structure, colors);//保存稀疏点云
	//以上为稀疏点云部分
	
	testImage(img_names, 8);//生成稠密点云（效果还不是很好。。而且很慢。。）
}