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
	cv::Mat img_left = res;
	cv::Mat img_right = ref;

	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout <<" 读取影像失败 "<< std::endl;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout <<" 左右影像尺寸不一致 "<< std::endl;
	}


	//···············································································//
	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// 左右影像的彩色数据
	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];
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
	printf("Done!\n");

	// PMS匹配参数设计
	PMSOption pms_option;
	// patch大小
	pms_option.patch_size = 15;
	// gamma
	pms_option.gamma = 10.0f;
	// alpha
	pms_option.alpha = 0.9f;
	// t_col
	pms_option.tau_col = 10.0f;
	// t_grad
	pms_option.tau_grad = 2.0f;
	// 传播迭代次数
	pms_option.num_iters = 4;

	// 一致性检查
	pms_option.is_check_lr = true;
	pms_option.lrcheck_thres = 1.0f;

	// 前端平行窗口
	pms_option.is_fource_fpw = false;


	// 定义PMS匹配类实例
	PatchMatchStereo pms;

	printf("PatchMatch Initializing...");
	auto start = std::chrono::steady_clock::now();
	//···············································································//
	// 初始化
	if (!pms.Initialize(width, height, pms_option)) {
		std::cout << " PMS初始化失败！" << std::endl;
		
	}
	auto end = std::chrono::steady_clock::now();
	auto tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	printf("PatchMatch Matching...");
	start = std::chrono::steady_clock::now();
	//···············································································//
	// 匹配
	// disparity数组保存子像素的视差结果
	auto disparity = new float32[uint32(width * height)]();
	if (!pms.Match(bytes_left, bytes_right, disparity)) {
		std::cout <<"PMS匹配失败! " << std::endl;
		
	}
	end = std::chrono::steady_clock::now();
	tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);
	
	// 保存点云
	SavePointCloud(bytes_left, pms.GetDisparityMap(0), width, height,0);

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

void SavePointCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height,int i)
{                                                                                                       //i是自己加的，感觉作用不是很大。。。                        
	// 不同图片，参数不一样，请修改下列参数值
	float32 B = 200.000;//基线
	float32 f = 1000.36;//焦距
	float32 x0l = i;		// 一图像主点x0
	float32 y0l = 0;	// 一图像主点y0
	float32 x0r = 350+i;// 二图像主点x0
	i = i + 50;

	std::vector<uint8> temp_img;
	for (sint32 y = 0; y < height; y) {
		for (sint32 x = 0; x < width; x) {
			const float32 disp = abs(disp_map[y * width + x]);
			if (disp == Invalid_Float) {
				continue;
			}
			float32 Z = B * f / (disp + (x0r - x0l));
			float32 X = Z * (x - x0l) / f;
			float32 Y = Z * (y - y0l) / f;
			// X Y Z R G B
			temp_img.push_back(X);
			temp_img.push_back(Y);
			temp_img.push_back(Z);
			temp_img.push_back(static_cast<int>(img_bytes[y * width * 3 + 3 * x + 2]));
			temp_img.push_back(static_cast<int>(img_bytes[y * width * 3 + 3 * x + 1]));
			temp_img.push_back(static_cast<int>(img_bytes[y * width * 3 + 3 * x]));
		}
	}


	// 手动输出点云ply文件
	std::ofstream ply_file("./result/densePoints.ply");
	// ply的头部信息
	ply_file << "ply\n";
	ply_file << "format ascii 1.0\n";
	ply_file << "element vertex " << temp_img.size()/6 << "\n";
	ply_file << "property float x\n";
	ply_file << "property float y\n";
	ply_file << "property float z\n";
	ply_file << "property uchar red\n";
	ply_file << "property uchar green\n";
	ply_file << "property uchar blue\n";
	ply_file << "end_header\n";
	// 写入点云数据 

	for (int i = 0; i < temp_img.size(); i += 6)
	{
		for (int j = 0; j < 5; ++j) ply_file << temp_img[i + j] << " ";
		ply_file << temp_img[i + 5] << "\n";
	}
	ply_file.close();
}