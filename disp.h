#pragma once
//#include "stdafx.h"
#include <iostream>
#include "pmstereo.h"
#include <chrono>
using namespace std::chrono;
using namespace cv;
#include <opencv2/opencv.hpp>
#include<fstream>

/*��ʾ�Ӳ�ͼ*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*�����Ӳ�ͼ*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*�����Ӳ����*/
void SavePointCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height,int i);

/**
* \brief
* \param argv 3
* \param argc argc[1]:��Ӱ��·�� argc[2]: ��Ӱ��·�� argc[3]: ��С�Ӳ�[��ѡ��Ĭ��0] argc[4]: ����Ӳ�[��ѡ��Ĭ��64]
* \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
* \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
* \return
*/
void dispmap(Mat res, Mat ref)
{



	cv::Mat img_left = res;
	cv::Mat img_right = ref;

	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
	
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
		
	}


	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// ����Ӱ��Ĳ�ɫ����
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

	// PMSƥ��������
	PMSOption pms_option;
	// patch��С
	pms_option.patch_size = 15;
	// gamma
	pms_option.gamma = 10.0f;
	// alpha
	pms_option.alpha = 0.9f;
	// t_col
	pms_option.tau_col = 10.0f;
	// t_grad
	pms_option.tau_grad = 2.0f;
	// ������������
	pms_option.num_iters = 4;

	// һ���Լ��
	pms_option.is_check_lr = true;
	pms_option.lrcheck_thres = 1.0f;

	// ǰ��ƽ�д���
	pms_option.is_fource_fpw = false;

	// �����Ӳ��
	pms_option.is_integer_disp = false;

	// ����PMSƥ����ʵ��
	PatchMatchStereo pms;

	printf("PatchMatch Initializing...");
	auto start = std::chrono::steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʼ��
	if (!pms.Initialize(width, height, pms_option)) {
		std::cout << "PMS��ʼ��ʧ�ܣ�" << std::endl;
		
	}
	auto end = std::chrono::steady_clock::now();
	auto tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	printf("PatchMatch Matching...");
	start = std::chrono::steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ƥ��
	// disparity���鱣�������ص��Ӳ���
	auto disparity = new float32[uint32(width * height)]();
	if (!pms.Match(bytes_left, bytes_right, disparity)) {
		std::cout << "PMSƥ��ʧ�ܣ�" << std::endl;
		
	}
	end = std::chrono::steady_clock::now();
	tt = duration_cast<std::chrono::milliseconds>(end - start);
	printf("Done! Timing : %lf s\n", tt.count() / 1000.0);



	
	//SaveDisparityMap(pms.GetDisparityMap(0), width, height, path_left);
	
	// �������
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
	// ��ʾ�Ӳ�ͼ
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
	// �����Ӳ�ͼ
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
{                                                                                                       //i���Լ��ӵģ��о����ò��Ǻܴ󡣡���                        
	// ��ͬͼƬ��������һ�������޸����в���ֵ
	float32 B = 200.000;//����
	float32 f = 1000.36;//����
	float32 x0l = i;		// һͼ������x0
	float32 y0l = 0;	// һͼ������y0
	float32 x0r = 350+i;// ��ͼ������x0
	i = i + 50;

	/*
	// �������
	FILE* fp_disp_cloud;
	fopen_s(&fp_disp_cloud, "OPcloud.txt", "a");
	if (fp_disp_cloud) {
		for (sint32 y = 0; y < height; y++) {
			for (sint32 x = 0; x < width; x++) {
				const float32 disp = abs(disp_map[y * width + x]);
				if (disp == Invalid_Float) {
					continue;
				}
				float32 Z = B * f / (disp + (x0r - x0l));
				float32 X = Z * (x - x0l) / f;
				float32 Y = Z * (y - y0l) / f;
				// X Y Z R G B
				fprintf_s(fp_disp_cloud, "%f %f %f %d %d %d\n", X, Y,
					Z, img_bytes[y * width * 3 + 3 * x + 2], img_bytes[y * width * 3 + 3 * x + 1], img_bytes[y * width * 3 + 3 * x]);
			}
		}
		fclose(fp_disp_cloud);
	}*/

	// �ֶ��������ply�ļ�
	std::ofstream ply_file("densePoints.ply");

	// ply��ͷ����Ϣ
	ply_file << "ply\n";
	ply_file << "format ascii 1.0\n";
	ply_file << "element vertex " << (height/3) * (width/3) << "\n";
	ply_file << "property float x\n";
	ply_file << "property float y\n";
	ply_file << "property float z\n";
	ply_file << "property uchar red\n";
	ply_file << "property uchar green\n";
	ply_file << "property uchar blue\n";
	ply_file << "end_header\n";
	// д���������
	for (sint32 y = 0; y < height; y+=3) {
		for (sint32 x = 0; x < width; x+=3) {
			const float32 disp = abs(disp_map[y * width + x]);
			if (disp == Invalid_Float) {
				continue;
			}
			float32 Z = B * f / (disp + (x0r - x0l));
			float32 X = Z * (x - x0l) / f;
			float32 Y = Z * (y - y0l) / f;
			// X Y Z R G B
			ply_file << X << " " << Y << " " << Z << " "
				<< static_cast<int>(img_bytes[y * width * 3 + 3 * x + 2]) << " "
				<< static_cast<int>(img_bytes[y * width * 3 + 3 * x + 1]) << " "
				<< static_cast<int>(img_bytes[y * width * 3 + 3 * x]) << std::endl;
		}
	}

	ply_file.close();

}