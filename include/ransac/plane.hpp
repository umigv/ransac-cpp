#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <random>
#include <array>
#include "ransac/types.hpp"

namespace ransac {

// Replace inf/nan with -1, clamp >10000 to -1.  Modifies in-place.
void clean_depths(cv::Mat& depths);

// Max-pool over non-overlapping (kh x kw) blocks
cv::Mat block_max_pool(const cv::Mat& src, int kh, int kw);

// Count inliers for plane z = c1*x + c2*y + c3 on pooled depth map
int metric(const cv::Mat& pooled, float c1, float c2, float c3, float tol);

// Return binary (0/1) mask where plane fits inv_depths within tol (squared)
cv::Mat make_mask(const cv::Mat& inv_depths, float c1, float c2, float c3, float tol);

// Sample 3 valid points from pooled and solve for plane coefficients
Eigen::Vector3f sample_and_solve(const cv::Mat& pooled, std::mt19937& rng);

// RANSAC ground plane fit.
// Returns {ground_mask (CV_8U 0/1), pixel-space coeffs / max_depth}
std::pair<cv::Mat, Eigen::Vector3f> ground_plane(
    const cv::Mat& depths, int iters, int kh, int kw, float tol,
    Eigen::Vector3f guess, std::mt19937& rng);

// White-pixel HSV mask (0 or 255)
cv::Mat hsv_mask(const cv::Mat& bgr_image);

// Ground plane RANSAC + HSV lane removal + morphology cleanup.
// Returns {driveable_mask (CV_8U 0/1), pixel-space coeffs / max_depth}
std::pair<cv::Mat, Eigen::Vector3f> hsv_and_ransac(
    const cv::Mat& bgr_image, const cv::Mat& depths,
    int iters, int kh, int kw, float tol, std::mt19937& rng);

// Convert pixel-space coefficients to real-world coefficients
std::array<float, 3> real_coeffs(const Eigen::Vector3f& px_coeffs,
                                  const Intrinsics& intr);

// Angle of depression (radians) from real-world coefficients
float real_angle(const std::array<float, 3>& rc);

}  // namespace ransac
