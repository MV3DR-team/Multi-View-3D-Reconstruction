#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <fstream>

const double MRT = 0.7;

/**
 *  两张图之间的特征提取与匹配
 */

void extract_features(
        const std::vector<cv::Mat>& images,
        std::vector<cv::Mat>& descriptors,
        std::vector<std::vector<cv::KeyPoint>>& keypoints,
        std::vector<std::vector<cv::Vec3b>>& colors)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    for (const auto& image : images)
    {
        std::vector<cv::KeyPoint> kp;
        cv::Mat des;
        std::vector<cv::Vec3b> color;

        sift->detectAndCompute(image, cv::noArray(), kp, des);
        keypoints.push_back(kp);
        descriptors.push_back(des);

        for(const auto& k : kp)
            color.push_back(image.at<cv::Vec3b>(k.pt));
        colors.push_back(color);
    }
}

void match_features(const cv::Mat& query, const cv::Mat& train, std::vector<cv::DMatch>& result)
{
    cv::Ptr<cv::FlannBasedMatcher> flann = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> matches;

    flann->knnMatch(query, train, matches, 2);

    // Lowe's SIFT matching ratio test(MRT)
    for (const auto& match : matches)
        if (match[0].distance < MRT * match[1].distance) result.push_back(match[0]);
}

void match_all_features(const std::vector<cv::Mat>& descriptors_all, std::vector<std::vector<cv::DMatch>>& matches_all)
{
    for (int i = 1; i < descriptors_all.size(); ++i)
    {
        std::vector<cv::DMatch> match;
        match_features(descriptors_all[i - 1],descriptors_all[i], match);
        matches_all.push_back(match);
    }
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

template<typename T_>
void maskout_points(std::vector<T_>& p, const cv::Mat& mask)
{
    std::vector<T_> temp;
    for (int64_t i = 0; i < mask.rows; ++i)
        if (mask.at<uchar>(i)) temp.push_back(p[i]);
    p.swap(temp);
}

void convert_to_2xN_mat(const std::vector<cv::Point2d>& p, cv::Mat& ph)
{
    for(int i = 0; i < p.size(); ++i)
        ph.at<double>(0, i) = p[i].x, ph.at<double>(1, i) = p[i].y;
}

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

/**
 * 点云融合
 */

void get_objpoints_and_imgpoints(
        const std::vector<cv::DMatch>& matches,
        const std::vector<int64_t>& structure_indices,
        const std::vector<cv::Point3d>& structure,
        const std::vector<cv::KeyPoint>& keypoints,
        std::vector<cv::Point3d>& object_points,
        std::vector<cv::Point2d>& image_points)
{
    for (const auto& match : matches)
    {
        int64_t struct_idx = structure_indices[match.queryIdx];
        if (struct_idx >= 0)
        {
            object_points.push_back(structure[struct_idx]);
            image_points.push_back(keypoints[match.trainIdx].pt);
        }
    }
}

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

/**
 * Bundle Adjustment
 */

struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d& observation) : observation(observation) {}

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

//void bundle_adjustment(
//        std::vector<cv::Mat>& rotations,
//        const std::vector<cv::Mat>& motions,
//        const cv::Mat& K,
//        const std::vector<std::vector<int64_t>>& correspond_struct_idx,
//        const std::vector<std::vector<cv::KeyPoint>>& keypoints_all,
//        std::vector<cv::Point3d>& structure)
//{
//    //todo
//}


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
        ply_file << structure[i].x << " " << structure[i].y << " " <<structure[i].z << " "
                 << static_cast<int>(colors[i][2]) << " "
                 << static_cast<int>(colors[i][1]) << " "
                 << static_cast<int>(colors[i][0]) << std::endl;
    }
    ply_file.close();
}

void save_pcd_file(
        const std::string& pcd_save_path,
        std::vector<cv::Point3d>& structure,
        const std::vector<cv::Vec3b>& colors
)
{
    // 手动输出点云ply文件
    std::ofstream pcd_file(pcd_save_path);

    // ply的头部信息
    pcd_file << "# .PCD v0.7 - Point Cloud Data file format\n";
    pcd_file << "VERSION 0.7\n";
    pcd_file << "FIELDS x y z rgb "  << "\n";
    pcd_file << "SIZE 4 4 4 4\n";
    pcd_file << "TYPE F F F F\n";
    pcd_file << "COUNT 1 1 1 1\n";
    pcd_file << "WIDTH "<< structure.size() * 4 << "\n";
    pcd_file << "HEIGHT 1\n";
    pcd_file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    pcd_file << "POINTS "<< structure.size() * 4 << "\n";
    pcd_file << "DATA ascii\n\n";
    // 写入点云数据
    for (int i = 0; i < structure.size(); ++i)
    {
        int value = (static_cast<int>(colors[i][2]) << 16) | (static_cast<int>(colors[i][1]) << 8) | static_cast<int>(colors[i][0]);
        pcd_file << structure[i].x << " " << structure[i].y << " " <<structure[i].z << " "
                 << value << std::endl;
    }
    pcd_file.close();
}



void bundle_adjustment(
       cv:: Mat& intrinsic,
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
        if(img_idx == 1)
        {
            std::cout<<"stop!!!";
        }
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;
            std::cout << img_idx<<":"<<point_idx<<std::endl;
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


int main()
{
    //std::freopen("noBa.txt", "w", stdout);
    std::vector<std::string> images_path;

    images_path.push_back("/home/maric/CLionProjects/reconstruction/s2/B21.jpg");
    images_path.push_back("/home/maric/CLionProjects/reconstruction/s2/B22.jpg");
    images_path.push_back("/home/maric/CLionProjects/reconstruction/s2/B23.jpg");
    images_path.push_back("/home/maric/CLionProjects/reconstruction/s2/B24.jpg");
    images_path.push_back("/home/maric/CLionProjects/reconstruction/s2/B25.jpg");

    std::vector<cv::Mat> images;
    for (const auto& image : images_path) images.push_back(cv::imread(image));

    // 相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
                                        719.5459, 0, 0,
            0, 719.5459, 0,
            0, 0, 1);

    std::vector<cv::Mat> descriptors_all;
    std::vector<std::vector<cv::KeyPoint>> keypoints_all;
    std::vector<std::vector<cv::Vec3b>> colors_all;
    std::vector<std::vector<cv::DMatch>> matches_all;

    extract_features(images, descriptors_all, keypoints_all, colors_all);
    match_all_features(descriptors_all, matches_all);

    std::vector<cv::Point3d> structure;
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> motions;
    std::vector<std::vector<int64_t>> correspond_struct_idx;
    std::vector<cv::Vec3b> colors;

    init_structure(K, keypoints_all, colors_all, matches_all, structure,
                   rotations, motions, correspond_struct_idx, colors);


    for (int i = 1; i < matches_all.size(); ++i)
    {
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        get_objpoints_and_imgpoints(matches_all[i], correspond_struct_idx[i],
                                    structure, keypoints_all[i + 1], object_points, image_points);

        cv::Mat r, R, T;
        cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), r, T);
        cv::Rodrigues(r, R);
        rotations.push_back(R);
        motions.push_back(T);

        std::vector<cv::Point2d> p1, p2;
        std::vector<cv::Vec3b> c1, c2;
        std::vector<cv::Point3d> next_structure;
        get_matched_points(keypoints_all[i], keypoints_all[i + 1], matches_all[i], p1, p2);
        get_matched_colors(colors_all[i], colors_all[i + 1], matches_all[i], c1, c2);
        reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);
        fusion_structure(matches_all[i], correspond_struct_idx[i], correspond_struct_idx[i + 1],
                         structure, next_structure, colors, c1);
    }

    std::vector<cv::Mat> extrinsics;
    cv::Mat intrinsic(cv::Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
    for (size_t i = 0; i < rotations.size(); ++i)
    {
        cv::Mat extrinsic(6, 1, CV_64FC1);
        cv::Mat r;
        Rodrigues(rotations[i], r);




        r.copyTo(extrinsic.rowRange(0, 3));
        motions[i].copyTo(extrinsic.rowRange(3, 6));

        extrinsics.push_back(extrinsic);


    }
    bundle_adjustment(intrinsic, extrinsics,correspond_struct_idx,keypoints_all,structure);



    for (int i = 0; i < structure.size(); ++i)
    {
        std::cout << structure[i].x << " " << structure[i].y << " " << -structure[i].z << " ";
        std::cout << (int)colors[i][2] << " " << (int)colors[i][1] << " " << (int)colors[i][0] << std::endl;
    }

    save_ply_file("/home/maric/CLionProjects/reconstruction/s2/output_test_BA.ply",structure,colors);
  //  save_pcd_file("/home/maric/CLionProjects/reconstruction/s2/output_test_BA.pcd",structure,colors);
    //bundle_adjustment(rotations, motions, K, correspond_struct_idx, keypoints_all, structure);
    return 0;

}

