//#include "stdafx.h"
#include "pmstereo.h"
#include <ctime>
#include <random>
#include "propagation.h"





PatchMatchStereo::PatchMatchStereo() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
gray_left_(nullptr), gray_right_(nullptr),
grad_left_(nullptr), grad_right_(nullptr),
cost_left_(nullptr), cost_right_(nullptr),
disp_left_(nullptr), disp_right_(nullptr),
plane_left_(nullptr), plane_right_(nullptr),
is_initialized_(false) { }

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



PatchMatchStereo::~PatchMatchStereo()
{
	Release();
}

bool PatchMatchStereo::Initialize(const sint32& width, const sint32& height, const PMSOption& option)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// PMS参数
	option_ = option;

	if (width <= 0 || height <= 0) {
		return false;
	}

	//··· 开辟内存空间
	const sint32 img_size = width * height;
	const sint32 disp_range = option.max_disparity - option.min_disparity;
	// 灰度数据
	gray_left_ = new uint8[img_size];
	gray_right_ = new uint8[img_size];
	// 梯度数据
	grad_left_ = new PGradient[img_size]();
	grad_right_ = new PGradient[img_size]();
	// 代价数据
	cost_left_ = new float32[img_size];
	cost_right_ = new float32[img_size];
	// 视差图
	disp_left_ = new float32[img_size];
	disp_right_ = new float32[img_size];
	// 平面集
	plane_left_ = new DisparityPlane[img_size];
	plane_right_ = new DisparityPlane[img_size];

	is_initialized_ = grad_left_ && grad_right_ && disp_left_ && disp_right_ && plane_left_ && plane_right_;

	return is_initialized_;
}



void PatchMatchStereo::Release()
{
	SAFE_DELETE(grad_left_);
	SAFE_DELETE(grad_right_);
	SAFE_DELETE(cost_left_);
	SAFE_DELETE(cost_right_);
	SAFE_DELETE(disp_left_);
	SAFE_DELETE(disp_right_);
	SAFE_DELETE(plane_left_);
	SAFE_DELETE(plane_right_);
}



bool PatchMatchStereo::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
	if (!is_initialized_) {
		return false;
	}
	if (img_left == nullptr || img_right == nullptr) {
		return false;
	}

	img_left_ = img_left;
	img_right_ = img_right;

	// 随机初始化
	RandomInitialization();

	// 计算灰度图
	ComputeGray();

	// 计算梯度图
	ComputeGradient();

	// 迭代传播
	Propagation();

	// 平面转换成视差
	PlaneToDisparity();

	// 左右一致性检查
	if (option_.is_check_lr) {
		// 一致性检查
		LRCheck();
	}

	// 输出视差图
	if (disp_left && disp_left_) {
		memcpy(disp_left, disp_left_, height_ * width_ * sizeof(float32));
	}
	return true;
}

bool PatchMatchStereo::Reset(const uint32& width, const uint32& height, const PMSOption& option)
{
	// 释放内存
	Release();

	// 重置初始化标记
	is_initialized_ = false;

	return Initialize(width, height, option);
}

float* PatchMatchStereo::GetDisparityMap(const sint32& view) const
{
	switch (view) {
	case 0:
		return disp_left_;
	case 1:
		return disp_right_;
	default:
		return nullptr;
	}
}

PGradient* PatchMatchStereo::GetGradientMap(const sint32& view) const
{
	switch (view) {
	case 0:
		return grad_left_;
	case 1:
		return grad_right_;
	default:
		return nullptr;
	}
}

void PatchMatchStereo::RandomInitialization() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr) {
		return;
	}
	const auto& option = option_;
	const sint32 min_disparity = option.min_disparity;
	const sint32 max_disparity = option.max_disparity;

	// 随机数生成器
	std::random_device rd;
	std::mt19937 gen(rd());
	const std::uniform_real_distribution<float32> rand_d(static_cast<float32>(min_disparity), static_cast<float32>(max_disparity));
	const std::uniform_real_distribution<float32> rand_n(-1.0f, 1.0f);
	std::exponential_distribution<> exp(1.0);

	for (int k = 0; k < 2; k++) {
		auto* disp_ptr = k == 0 ? disp_left_ : disp_right_;
		auto* plane_ptr = k == 0 ? plane_left_ : plane_right_;
		sint32 sign = (k == 0) ? 1 : -1;
		for (sint32 y = 0; y < height; y++) {
			for (sint32 x = 0; x < width; x++) {
				const sint32 p = y * width + x;
				// 随机视差值
				float32 disp = sign * exp(gen);
				if (option.is_integer_disp) {
					disp = static_cast<float32>(round(disp));
				}
				disp_ptr[p] = disp;

				// 随机法向量
				PVector3f norm;
				if (!option.is_fource_fpw) {
					norm.x = exp(gen);
					norm.y = exp(gen);
					float32 z = exp(gen);
					while (z == 0.0f) {
						z = exp(gen);
					}
					norm.z = z;
					norm.normalize();
				}
				else {
					norm.x = 0.0f; norm.y = 0.0f; norm.z = 1.0f;
				}

				// 计算视差平面
				plane_ptr[p] = DisparityPlane(x, y, norm, disp);
			}
		}
	}
}

void PatchMatchStereo::ComputeGray() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		img_left_ == nullptr || img_right_ == nullptr ||
		gray_left_ == nullptr || gray_right_ == nullptr) {
		return;
	}

	// 彩色转灰度
	for (sint32 n = 0; n < 2; n++) {
		auto* color = (n == 0) ? img_left_ : img_right_;
		auto* gray = (n == 0) ? gray_left_ : gray_right_;
		for (sint32 i = 0; i < height; i++) {
			for (sint32 j = 0; j < width; j++) {
				const auto b = color[i * width * 3 + 3 * j];
				const auto g = color[i * width * 3 + 3 * j + 1];
				const auto r = color[i * width * 3 + 3 * j + 2];
				gray[i * width + j] = uint8(r * 0.299 + g * 0.587 + b * 0.114);
			}
		}
	}
}

void PatchMatchStereo::ComputeGradient() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		grad_left_ == nullptr || grad_right_ == nullptr ||
		gray_left_ == nullptr || gray_right_ == nullptr) {
		return;
	}

	// Sobel梯度算子
	for (sint32 n = 0; n < 2; n++) {
		auto* gray = (n == 0) ? gray_left_ : gray_right_;
		auto* grad = (n == 0) ? grad_left_ : grad_right_;
		for (int y = 1; y < height - 1; y++) {
			for (int x = 1; x < width - 1; x++) {
				const auto grad_x = (-gray[(y - 1) * width + x - 1] + gray[(y - 1) * width + x + 1]) +
					(-2 * gray[y * width + x - 1] + 2 * gray[y * width + x + 1]) +
					(-gray[(y + 1) * width + x - 1] + gray[(y + 1) * width + x + 1]);
				const auto grad_y = (-gray[(y - 1) * width + x - 1] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + x + 1]) +
					(gray[(y + 1) * width + x - 1] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + x + 1]);

				// 这里除以8是为了让梯度的最大值不超过255，这样计算代价时梯度差和颜色差位于同一个尺度
				grad[y * width + x].x = grad_x / 8;
				grad[y * width + x].y = grad_y / 8;
			}
		}
	}
}

void PatchMatchStereo::Propagation() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		img_left_ == nullptr || img_right_ == nullptr ||
		grad_left_ == nullptr || grad_right_ == nullptr ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr) {
		return;
	}

	// 左右视图匹配参数
	const auto opion_left = option_;
	auto option_right = option_;
	option_right.min_disparity = -opion_left.max_disparity;
	option_right.max_disparity = -opion_left.min_disparity;

	// 左右视图传播实例
	PMSPropagation propa_left(width, height, img_left_, img_right_, grad_left_, grad_right_, plane_left_, plane_right_, opion_left, cost_left_, cost_right_, disp_left_);
	PMSPropagation propa_right(width, height, img_right_, img_left_, grad_right_, grad_left_, plane_right_, plane_left_, option_right, cost_right_, cost_left_, disp_right_);

	// 迭代传播
	for (int k = 0; k < option_.num_iters; k++) {
		propa_left.DoPropagation();
		propa_right.DoPropagation();
	}
}

void PatchMatchStereo::LRCheck()
{
	const sint32 width = width_;
	const sint32 height = height_;

	const float32& threshold = option_.lrcheck_thres;

	// k==0 : 左视图一致性检查
	// k==1 : 右视图一致性检查
	for (int k = 0; k < 2; k++) {
		auto* disp_left = (k == 0) ? disp_left_ : disp_right_;
		auto* disp_right = (k == 0) ? disp_right_ : disp_left_;
		auto& mismatches = (k == 0) ? mismatches_left_ : mismatches_right_;
		mismatches.clear();

		// ---左右一致性检查
		for (sint32 y = 0; y < height; y++) {
			for (sint32 x = 0; x < width; x++) {

				// 左影像视差值
				auto& disp = disp_left[y * width + x];

				if (disp == Invalid_Float) {
					mismatches.emplace_back(x, y);
					continue;
				}

				// 根据视差值找到右影像上对应的同名像素
				const auto col_right = lround(x - disp);

				if (col_right >= 0 && col_right < width) {
					// 右影像上同名像素的视差值
					auto& disp_r = disp_right[y * width + col_right];

					// 判断两个视差值是否一致（差值在阈值内为一致）
					// 在本代码里，左右视图的视差值符号相反
					if (abs(disp + disp_r) > threshold) {
						// 让视差值无效
						disp = Invalid_Float;
						mismatches.emplace_back(x, y);
					}
				}
				else {
					// 通过视差值在右影像上找不到同名像素（超出影像范围）
					disp = Invalid_Float;
					mismatches.emplace_back(x, y);
				}
			}
		}
	}
}

void PatchMatchStereo::PlaneToDisparity() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr) {
		return;
	}
	for (int k = 0; k < 2; k++) {
		auto* plane_ptr = (k == 0) ? plane_left_ : plane_right_;
		auto* disp_ptr = (k == 0) ? disp_left_ : disp_right_;
		for (sint32 y = 0; y < height; y++) {
			for (sint32 x = 0; x < width; x++) {
				const sint32 p = y * width + x;
				const auto& plane = plane_ptr[p];
				disp_ptr[p] = plane.to_disparity(x, y);
			}
		}
	}
}
