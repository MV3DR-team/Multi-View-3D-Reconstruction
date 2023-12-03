#pragma once
#include <iostream>
#include "pmstereo.h"
#include <chrono>
using namespace std::chrono;
using namespace cv;
#include <opencv2/opencv.hpp>
#include<fstream>

/*显示视差图*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*保存视差图*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*保存视差点云*/
void SavePointCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height,int i);

/**
* \brief
* \param argv 3
* \param argc argc[1]:左影像路径 argc[2]: 右影像路径 argc[3]: 最小视差[可选，默认0] argc[4]: 最大视差[可选，默认64]
* \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
* \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
* \return
*/
void dispmap(Mat res, Mat ref)
{
	// 将传入的Mat对象分别赋值给img_left和img_right
	cv::Mat img_left = res;
	cv::Mat img_right = ref;

	// 检查图像是否成功读取
	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << " 读取影像失败 " << std::endl;
	}
	// 检查左右影像尺寸是否一致
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << " 左右影像尺寸不一致 " << std::endl;
	}

	// 获取图像的宽度和高度
	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// 创建用于存储图像数据的字节数组
	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];

	// 将图像数据存储到字节数组中
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}

	// 定义PMSOption结构体并设置其参数值
	PMSOption pms_option;
	pms_option.patch_size = 15;
	pms_option.gamma = 10.0f;
	pms_option.alpha = 0.9f;
	pms_option.tau_col = 10.0f;
	pms_option.tau_grad = 2.0f;
	pms_option.num_iters = 8;
	pms_option.is_check_lr = true;
	pms_option.lrcheck_thres = 1.0f;
	pms_option.is_fource_fpw = false;

	// 创建PatchMatchStereo对象
	PatchMatchStereo pms;

	// 初始化PMS对象，并计算初始化的时间
	auto start = std::chrono::steady_clock::now();
	if (!pms.Initialize(width, height, pms_option)) {
		std::cout << " PMS初始化失败！ " << std::endl;
	}
	auto end = std::chrono::steady_clock::now();
	auto tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	// 进行视差计算，并记录计算时间
	start = std::chrono::steady_clock::now();
	auto disparity = new float32[uint32(width * height)]();
	if (!pms.Match(bytes_left, bytes_right, disparity)) {
		std::cout << " PMS匹配失败! " << std::endl;
	}
	end = std::chrono::steady_clock::now();
	tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	// 保存点云数据
	SavePointCloud(bytes_left, pms.GetDisparityMap(0), width, height, 0);

	// 释放内存
	delete[] disparity;
	disparity = nullptr;
	delete[] bytes_left;
	bytes_left = nullptr;
	delete[] bytes_right;
	bytes_right = nullptr;
}

void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name)
{
	// 显示视差图
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(name, disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imshow(name + "-color", disp_color);

}

void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// 保存视差图
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imwrite(path + "-d.png", disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imwrite(path + "-c.png", disp_color);
}

void SavePointCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, int image_main_point)
{
	// 常量用于点云生成
	float32 B = 200.000; // 基线距离
	float32 f = 1000.36; // 焦距
	float32 x0l = image_main_point; // 左图像主点
	float32 y0l = 0; // 左图像垂直主点
	float32 x0r = 350 + image_main_point; // 右图像主点

	// 打印状态消息
	std::cout << " 正在生成稠密点云... " << std::endl;

	// 临时向量用于存储点的位置和颜色
	std::vector<float32> temp_pos;
	std::vector<int> temp_color;

	// 遍历图像中的每个像素
	for (sint32 y = 0; y < height; y++) {
		for (sint32 x = 0; x < width; x++) {
			// 计算当前像素的视差值
			const float32 disp = abs(disp_map[y * width + x]);

			// 检查视差值是否有效
			if (disp == Invalid_Float) {
				continue; // 跳过无效的视差值
			}

			// 使用立体几何计算点的三维坐标
			float32 Z = B * f / (disp + (x0r - x0l));
			float32 Y = Z * (y - y0l) / f;
			float32 X = Z * (x - x0l) / f;

			// 将点的三维位置和颜色存储在临时向量中
			temp_pos.push_back(X);
			temp_pos.push_back(Y);
			temp_pos.push_back(Z);
			temp_color.push_back(static_cast<sint32>(img_bytes[y * width * 3 + 3 * x + 2])); // 红色分量
			temp_color.push_back(static_cast<sint32>(img_bytes[y * width * 3 + 3 * x + 1])); // 绿色分量
			temp_color.push_back(static_cast<sint32>(img_bytes[y * width * 3 + 3 * x])); // 蓝色分量
		}
		std::system("cls");
		std::cout << " 正在生成点云文件： " << (y + 1) * width << "/" << height * width << std::endl;
		std::cout << "[";
		for (int i = 0; i < 100; i++)
			std::cout << ((i < 100 * (float)y / height) ? "=" : " ");
		std::cout << "]";
	}

	// 打印消息，保存稠密点云为PLY文件
	std::cout << " 正在将稠密点云保存为PLY文件...  " << std::endl;
	std::ofstream ply_file("./result/densePoints.ply");
	ply_file << "ply\n";
	ply_file << "format ascii 1.0\n";
	ply_file << "element vertex " << temp_pos.size() / 3 << "\n";
	ply_file << "property float x\n";
	ply_file << "property float y\n";
	ply_file << "property float z\n";
	ply_file << "property uchar red\n";
	ply_file << "property uchar green\n";
	ply_file << "property uchar blue\n";
	ply_file << "end_header\n";

	// 遍历临时向量中的点，将点的位置和颜色写入PLY文件
	for (int i = 0; i < temp_pos.size(); i += 3)
	{
		ply_file << temp_pos[i] << " " << temp_pos[i + 1] << " " << temp_pos[i + 2] << " "
			<< temp_color[i] << " " << temp_color[i + 1] << " " << temp_color[i + 2] << std::endl;
	}

	// 关闭PLY文件，打印保存成功的消息
	ply_file.close();
	std::cout << " 稠密点云已成功保存为PLY文件！程序结束 " << std::endl;
}