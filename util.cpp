#pragma once
//#include "stdafx.h"
#include "struct.h"
#include <vector>
#include <algorithm>


namespace pms_util
{

	/**
	* \brief 获取像素(i,j)的颜色值
	* \param img_data	颜色数组
	* \param width		影像宽
	* \param height		影像高
	* \param i			像素行坐标
	* \param j			像素列坐标
	* \return 像素(i,j)的颜色值
	*/
	PColor GetColor(const uint8* img_data, const sint32& width, const sint32& height, const sint32& i, const sint32& j);

	/**
	 * \brief 中值滤波
	 * \param in				输入，源数据
	 * \param out				输出，目标数据
	 * \param width				输入，宽度
	 * \param height			输入，高度
	 * \param wnd_size			输入，窗口宽度
	 */
	void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);

	/**
	 * \brief 加权中值滤波
	 * \param img_data		颜色数组
	 * \param width			影像宽
	 * \param height		影像高
	 * \param wnd_size		窗口大小
	 * \param gamma			gamma值
	 * \param filter_pixels 需要滤波的像素集
	 * \param disparity_map 视差图
	 */
	void WeightedMedianFilter(const uint8* img_data, const sint32& width, const sint32& height, const sint32& wnd_size, const float32& gamma, const vector<pair<int, int>>& filter_pixels, float32* disparity_map);

}

PColor pms_util::GetColor(const uint8* img_data, const sint32& width, const sint32& height, const sint32& x, const sint32& y)
{
	auto* pixel = img_data + y * width * 3 + 3 * x;
	return { pixel[0], pixel[1], pixel[2] };
}

void pms_util::MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height,
	const sint32 wnd_size)
{
	const sint32 radius = wnd_size / 2;
	const sint32 size = wnd_size * wnd_size;

	// 存储局部窗口内的数据
	std::vector<float32> wnd_data;
	wnd_data.reserve(size);

	for (sint32 y = 0; y < height; y++) {
		for (sint32 x = 0; x < width; x++) {
			wnd_data.clear();

			// 获取局部窗口数据
			for (sint32 r = -radius; r <= radius; r++) {
				for (sint32 c = -radius; c <= radius; c++) {
					const sint32 row = y + r;
					const sint32 col = x + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}

			// 排序
			std::sort(wnd_data.begin(), wnd_data.end());

			if (!wnd_data.empty()) {
				// 取中值
				out[y * width + x] = wnd_data[wnd_data.size() / 2];
			}
		}
	}
}


void pms_util::WeightedMedianFilter(const uint8* img_data, const sint32& width, const sint32& height, const sint32& wnd_size, const float32& gamma, const vector<pair<int, int>>& filter_pixels, float32* disparity_map)
{
	const sint32 wnd_size2 = wnd_size / 2;

	// 带权视差集
	vector<pair<float32, float32>> disps;
	disps.reserve(wnd_size * wnd_size);

	for (auto& pix : filter_pixels) {
		const sint32 x = pix.first;
		const sint32 y = pix.second;
		// weighted median filter
		disps.clear();
		const auto& col_p = GetColor(img_data, width, height, x, y);
		float32 total_w = 0.0f;
		for (sint32 r = -wnd_size2; r <= wnd_size2; r++) {
			for (sint32 c = -wnd_size2; c <= wnd_size2; c++) {
				const sint32 yr = y + r;
				const sint32 xc = x + c;
				if (yr < 0 || yr >= height || xc < 0 || xc >= width) {
					continue;
				}
				const auto& disp = disparity_map[yr * width + xc];
				if (disp == Invalid_Float) {
					continue;
				}
				// 计算权值
				const auto& col_q = GetColor(img_data, width, height, xc, yr);
				const auto dc = abs(col_p.r - col_q.r) + abs(col_p.g - col_q.g) + abs(col_p.b - col_q.b);
				const auto w = exp(-dc / gamma);
				total_w += w;

				// 存储带权视差
				disps.emplace_back(disp, w);
			}
		}

		// --- 取加权中值
		// 按视差值排序
		std::sort(disps.begin(), disps.end());
		const float32 median_w = total_w / 2;
		float32 w = 0.0f;
		for (auto& wd : disps) {
			w += wd.second;
			if (w >= median_w) {
				disparity_map[y * width + x] = wd.first;
				break;
			}
		}
	}
}