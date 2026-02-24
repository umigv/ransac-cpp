#include "ransac/plane.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ransac {

void clean_depths(cv::Mat& depths) {
    // Matches Python: replace inf/nan with -1, clamp >10000 to 10000
    for (int r = 0; r < depths.rows; r++) {
        float* row = depths.ptr<float>(r);
        for (int c = 0; c < depths.cols; c++) {
            float v = row[c];
            if (std::isinf(v) || std::isnan(v))
                row[c] = -1.f;
            else if (v > 10000.f)
                row[c] = 10000.f;
        }
    }
}

cv::Mat block_max_pool(const cv::Mat& src, int kh, int kw) {
    int oh = src.rows / kh;
    int ow = src.cols / kw;
    cv::Mat dst(oh, ow, CV_32F);
    for (int r = 0; r < oh; r++) {
        float* out = dst.ptr<float>(r);
        for (int c = 0; c < ow; c++) {
            float maxval = -std::numeric_limits<float>::infinity();
            for (int dr = 0; dr < kh; dr++) {
                const float* src_row = src.ptr<float>(r * kh + dr);
                for (int dc = 0; dc < kw; dc++)
                    maxval = std::max(maxval, src_row[c * kw + dc]);
            }
            out[c] = maxval;
        }
    }
    return dst;
}

// Branchless inner loop — auto-vectorises with -O3 -march=native
int metric(const cv::Mat& pooled, float c1, float c2, float c3, float tol) {
    int count = 0;
    for (int r = 0; r < pooled.rows; r++) {
        const float* row = pooled.ptr<float>(r);
        float row_base = c2 * r + c3;
        for (int c = 0; c < pooled.cols; c++) {
            float d   = row[c];
            float err = std::abs(row_base + c1 * c - d);
            count += static_cast<int>((d > 0.f) & (err < tol));
        }
    }
    return count;
}

// Matches Python mask(): tolerance is applied to err^2 (not |err|)
cv::Mat make_mask(const cv::Mat& inv_depths, float c1, float c2, float c3, float tol) {
    cv::Mat result(inv_depths.size(), CV_8U);
    for (int r = 0; r < inv_depths.rows; r++) {
        const float* src = inv_depths.ptr<float>(r);
        uint8_t*     dst = result.ptr<uint8_t>(r);
        float row_base = c2 * r + c3;
        for (int c = 0; c < inv_depths.cols; c++) {
            float d   = src[c];
            float err = row_base + c1 * c - d;
            dst[c] = static_cast<uint8_t>((d > 0.f) & (err * err < tol));
        }
    }
    return result;
}

Eigen::Vector3f sample_and_solve(const cv::Mat& pooled, std::mt19937& rng) {
    const int h = pooled.rows, w = pooled.cols;
    std::uniform_int_distribution<int> rdist(0, h - 1);
    std::uniform_int_distribution<int> cdist(0, w - 1);

    Eigen::Matrix3f A;
    Eigen::Vector3f b;
    while (true) {
        for (int i = 0; i < 3; i++) {
            int row, col;
            float d;
            do {
                row = rdist(rng);
                col = cdist(rng);
                d   = pooled.at<float>(row, col);
            } while (d < 0.f);
            A.row(i) << static_cast<float>(col), static_cast<float>(row), 1.f;
            b(i) = d;
        }
        if (A.fullPivLu().rank() == 3)
            break;
    }
    return A.fullPivLu().solve(b);
}

std::pair<cv::Mat, Eigen::Vector3f> ground_plane(
    const cv::Mat& depths_in, int iters, int kh, int kw, float tol,
    Eigen::Vector3f guess, std::mt19937& rng)
{
    cv::Mat depths = depths_in.clone();
    clean_depths(depths);

    double max_depth_d = 0.0;
    cv::minMaxLoc(depths, nullptr, &max_depth_d);
    const float max_depth = static_cast<float>(max_depth_d);

    // inv_depths = max_depth / depths  (invalid pixels → negative, filtered by metric/mask)
    cv::Mat inv_depths;
    cv::divide(static_cast<double>(max_depth), depths, inv_depths);

    cv::Mat pooled = block_max_pool(inv_depths, kh, kw);

    // Scale guess into pooled coordinate space
    Eigen::Vector3f best_coeffs = guess;
    best_coeffs(0) *= max_depth * kw;
    best_coeffs(1) *= max_depth * kh;
    best_coeffs(2) *= max_depth;
    int best_score = metric(pooled, best_coeffs(0), best_coeffs(1), best_coeffs(2), tol);

    for (int i = 0; i < iters; i++) {
        Eigen::Vector3f coeffs = sample_and_solve(pooled, rng);
        int score = metric(pooled, coeffs(0), coeffs(1), coeffs(2), tol);
        if (score > best_score) {
            best_score  = score;
            best_coeffs = coeffs;
        }
    }

    // Convert from pooled to original pixel space
    best_coeffs(0) /= kw;
    best_coeffs(1) /= kh;

    cv::Mat result = make_mask(inv_depths, best_coeffs(0), best_coeffs(1), best_coeffs(2), tol);

    return {result, best_coeffs / max_depth};
}

cv::Mat hsv_mask(const cv::Mat& bgr_image) {
    cv::Mat hsv;
    cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 190), cv::Scalar(255, 50, 255), mask);
    return mask;  // 0 or 255
}

std::pair<cv::Mat, Eigen::Vector3f> hsv_and_ransac(
    const cv::Mat& bgr_image, const cv::Mat& depths,
    int iters, int kh, int kw, float tol, std::mt19937& rng)
{
    auto [ground_mask, coeffs] = ground_plane(
        depths, iters, kh, kw, tol, Eigen::Vector3f::Zero(), rng);

    cv::Mat hsv_bin = hsv_mask(bgr_image);  // 0 or 255

    // driveable = ground AND NOT white-lane-pixel
    cv::Mat driveable(ground_mask.size(), CV_8U);
    for (int r = 0; r < ground_mask.rows; r++) {
        const uint8_t* g   = ground_mask.ptr<uint8_t>(r);
        const uint8_t* h   = hsv_bin.ptr<uint8_t>(r);
        uint8_t*       out = driveable.ptr<uint8_t>(r);
        for (int c = 0; c < ground_mask.cols; c++)
            out[c] = (g[c] != 0 && h[c] == 0) ? 255 : 0;
    }

    cv::Mat close_kernel = cv::Mat::ones(2, 2, CV_8U);
    cv::morphologyEx(driveable, driveable, cv::MORPH_CLOSE, close_kernel);
    cv::Mat open_kernel = cv::Mat::ones(7, 7, CV_8U);
    cv::morphologyEx(driveable, driveable, cv::MORPH_OPEN, open_kernel);

    // Convert to 0/1 to match Python's .astype(bool)
    cv::threshold(driveable, driveable, 0, 1, cv::THRESH_BINARY);

    return {driveable, coeffs};
}

std::array<float, 3> real_coeffs(const Eigen::Vector3f& px_c, const Intrinsics& intr) {
    const float c1 = px_c(0), c2 = px_c(1), c3 = px_c(2);
    const float d  = 1.f / (c1 * intr.cx + c2 * intr.cy + c3);
    return {-d * c1 * intr.fx, d * c2 * intr.fy, d};
}

float real_angle(const std::array<float, 3>& rc) {
    const float a = rc[0], b = rc[1];
    const float r = std::acos(1.f / std::hypot(a, b, 1.f));
    return std::isnan(r) ? 0.f : static_cast<float>(M_PI_2) - r;
}

}  // namespace ransac
