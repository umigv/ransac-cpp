#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include "ransac/types.hpp"

namespace ransac {

// Collect non-zero pixels from mask into an Nx3 point cloud (col, row, depth).
// Every (1+skip)-th point is kept.
PointCloud create_point_cloud(const cv::Mat& mask, const cv::Mat& depths, int skip = 3);

// Unproject pixel cloud to real-world (x, y, z) and apply pitch + orientation rotation.
// orientation (radians): positive rotates camera left
PointCloud pixel_to_real(const PointCloud& px_cloud,
                          const std::array<float, 3>& rc,
                          const Intrinsics& intr,
                          float orientation = 0.f);

// Bin real-world point cloud (x, z columns) into a grid. Returns CV_8U 0/1.
cv::Mat occupancy_grid(const PointCloud& real_pc, const GridConfiguration& conf);

// Combine driveable and blocked grids into a composite (0=blocked, 127=unknown, 255=driveable)
cv::Mat composite(const cv::Mat& drive_occ, const cv::Mat& block_occ);

// Propagate known cells along lines of sight from each VirtualCamera.
// Modifies merged in-place and returns it.
cv::Mat create_los_grid(cv::Mat merged, const std::vector<VirtualCamera>& cameras);

}  // namespace ransac
