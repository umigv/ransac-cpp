#pragma once
#include <Eigen/Dense>

namespace ransac {

struct Intrinsics {
    float cx, cy, fx, fy;
};

struct GridConfiguration {
    float gw, gh, cw;
    int   thres;
};

struct VirtualCamera {
    int   i, j;
    float dir;  // radians
    float fov;  // radians
};

// N×3 row-major matrix: each row is (x, y, z) or (col, row, depth)
using PointCloud = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

}  // namespace ransac
