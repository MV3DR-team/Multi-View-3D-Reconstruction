#pragma once
#include "pch.h"

/** \brief 基础类型别名 */
typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

/** \brief float32无效值 */


namespace metric {

	using namespace cv;
	using namespace std;

	constexpr double uc2f = 1.0 / 255.0;

	double sim_abs_diff(Mat& a, Mat& b, int _) {
		Mat tmp;
		absdiff(a, b, tmp);
		return -sum(sum(tmp))[0];
	}

	template <typename T>
	double sim_diff_dxy(Mat& a, Mat& b, int _) {
		double ndiff = sim_abs_diff(a, b, _);
		double dxy = 0.0;
		for (int i = 1; i < a.rows - 1; i++)
		{
			for (int j = 1; j < a.cols - 1; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					dxy += abs((a.at<T>(i + 1, j)[k] * uc2f - a.at<T>(i - 1, j)[k] * uc2f)
						- (b.at<T>(i + 1, j)[k] * uc2f - b.at<T>(i - 1, j)[k] * uc2f)) * 0.5;
					dxy += abs((a.at<T>(i, j + 1)[k] * uc2f - a.at<T>(i, j - 1)[k] * uc2f)
						- (b.at<T>(i, j + 1)[k] * uc2f - b.at<T>(i, j - 1)[k] * uc2f)) * 0.5;
				}
			}
		}
		return ndiff - dxy * 255.0 / 6.0;
	}

	double sigma(Mat& m, int i, int j, int block_size) {
		double sd = 0;

		Mat m_tmp = m(Range(i, i + block_size), Range(j, j + block_size));
		Mat m_squared(block_size, block_size, CV_64F);

		multiply(m_tmp, m_tmp, m_squared);

		double avg = mean(m_tmp)[0];
		double avg_2 = mean(m_squared)[0];

		sd = sqrt(avg_2 - avg * avg);

		return sd;
	}

	double cov(Mat& m1, Mat& m2, int i, int j, int block_size) {
		Mat m3 = Mat::zeros(block_size, block_size, m1.depth());
		Mat m1_tmp = m1(Range(i, i + block_size), Range(j, j + block_size));
		Mat m2_tmp = m2(Range(i, i + block_size), Range(j, j + block_size));

		multiply(m1_tmp, m2_tmp, m3);

		double avg_ro = mean(m3)[0];		// E(XY)
		double avg_r = mean(m1_tmp)[0];		// E(X)
		double avg_o = mean(m2_tmp)[0];		// E(Y)

		double sd_ro = avg_ro - avg_o * avg_r;	// E(XY) - E(X)E(Y)

		return sd_ro;
	}

	double ssim(Mat& img_src, Mat& img_compressed, int block_size) {
		double ssim = 0;
		int nBlockPerHeight = img_src.rows / block_size;
		int nBlockPerWidth = img_src.cols / block_size;

		double C1 = 0.01, C2 = 0.03;

		for (int i = 0; i < nBlockPerHeight; i++) {
			for (int j = 0; j < nBlockPerWidth; j++) {
				int m = i * block_size, n = j * block_size;

				double avg_o = mean(img_src(Range(i, i + block_size), Range(j, j + block_size)))[0];
				double avg_r = mean(img_compressed(Range(i, i + block_size), Range(j, j + block_size)))[0];
				double sigma_o = sigma(img_src, m, n, block_size);
				double sigma_r = sigma(img_compressed, m, n, block_size);
				double sigma_ro = cov(img_src, img_compressed, m, n, block_size);

				ssim += ((2 * avg_o * avg_r * C1) * (2 * sigma_ro + C2))
					/ ((avg_o * avg_o + avg_r * avg_r + C1) * (sigma_o * sigma_o + sigma_r * sigma_r + C2));
			}
		}

		ssim /= nBlockPerHeight * nBlockPerWidth;

		return ssim;
	}
}
