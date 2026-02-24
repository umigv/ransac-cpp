#include "ransac/occu.hpp"
#include "ransac/plane.hpp"
#include <cmath>
#include <algorithm>

namespace ransac {

PointCloud create_point_cloud(const cv::Mat& mask, const cv::Mat& depths, int skip) {
    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);  // pt.x = col, pt.y = row

    const int step = 1 + skip;
    const int n    = ((int)pts.size() + step - 1) / step;
    PointCloud cloud(n, 3);

    int idx = 0;
    for (int i = 0; i < (int)pts.size(); i += step) {
        const int x = pts[i].x;  // col
        const int y = pts[i].y;  // row
        cloud(idx, 0) = static_cast<float>(x);
        cloud(idx, 1) = static_cast<float>(y);
        cloud(idx, 2) = depths.at<float>(y, x);
        idx++;
    }
    return cloud.topRows(idx);
}

PointCloud pixel_to_real(const PointCloud& px_cloud,
                          const std::array<float, 3>& rc,
                          const Intrinsics& intr,
                          float orientation)
{
    PointCloud cloud = px_cloud;

    // Unproject: (col, row, depth) → (x_cam, y_cam, depth)
    cloud.col(0) = px_cloud.col(2).array() * (px_cloud.col(0).array() - intr.cx) / intr.fx;
    cloud.col(1) = px_cloud.col(2).array() * (intr.cy - px_cloud.col(1).array()) / intr.fy;
    // col 2 (depth) unchanged

    const float dep = real_angle(rc);
    const float c1  = std::cos(dep),         s1 = std::sin(dep);
    const float c2  = std::cos(orientation), s2 = std::sin(orientation);

    // Matches Python:
    //   R = Rx.transpose() @ Ry.transpose()
    //   result = cloud @ R
    // where Rx rotates around X (depression), Ry rotates around Y (orientation)
    Eigen::Matrix3f Rx, Ry;
    Rx << 1,  0,   0,
          0, c1, -s1,
          0, s1,  c1;
    Ry << c2,  0, -s2,
          0,   1,   0,
          s2,  0,  c2;

    const Eigen::Matrix3f R = (Ry * Rx).transpose();
    return cloud * R;
}

cv::Mat occupancy_grid(const PointCloud& real_pc, const GridConfiguration& conf) {
    const int W = static_cast<int>(conf.gw / conf.cw);
    const int H = static_cast<int>(conf.gh / conf.cw);

    cv::Mat counts = cv::Mat::zeros(H, W, CV_32S);

    for (int i = 0; i < real_pc.rows(); i++) {
        const int gx = W / 2 + static_cast<int>(real_pc(i, 0) / conf.cw);  // x → col
        const int gy = H - 1 - static_cast<int>(real_pc(i, 2) / conf.cw);  // z → row
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            counts.at<int>(gy, gx)++;
    }

    // Return 0/1 mask: 1 where count >= thres
    cv::Mat result;
    cv::compare(counts, conf.thres, result, cv::CMP_GE);  // 0 or 255
    cv::threshold(result, result, 0, 1, cv::THRESH_BINARY);
    return result;
}

cv::Mat composite(const cv::Mat& drive_occ, const cv::Mat& block_occ) {
    cv::Mat out(drive_occ.size(), CV_8U);
    for (int r = 0; r < drive_occ.rows; r++) {
        const uint8_t* d   = drive_occ.ptr<uint8_t>(r);
        const uint8_t* b   = block_occ.ptr<uint8_t>(r);
        uint8_t*       dst = out.ptr<uint8_t>(r);
        for (int c = 0; c < drive_occ.cols; c++) {
            const bool is_drive = d[c] != 0;
            const bool is_block = b[c] != 0;
            if      ( is_drive && !is_block) dst[c] = 255;  // driveable
            else if (!is_drive && !is_block) dst[c] = 127;  // unknown
            else                             dst[c] = 0;    // blocked
        }
    }
    return out;
}

cv::Mat create_los_grid(cv::Mat merged, const std::vector<VirtualCamera>& cameras) {
    const int h = merged.rows, w = merged.cols;

    for (const auto& cam : cameras) {
        const float dx0 =  std::cos(cam.dir - cam.fov / 2.f);
        const float dy0 = -std::sin(cam.dir - cam.fov / 2.f);
        const float dx1 =  std::cos(cam.dir + cam.fov / 2.f);
        const float dy1 = -std::sin(cam.dir + cam.fov / 2.f);

        const float ray_len = 2.f * (h + w);
        float fx0 = cam.j + dx0 * ray_len, fy0 = cam.i + dy0 * ray_len;
        float fx1 = cam.j + dx1 * ray_len, fy1 = cam.i + dy1 * ray_len;

        // Clip to image bounds, adjusting the other axis proportionally
        const float nx0 = std::clamp(fx0, 0.f, (float)(w - 1));
        const float nx1 = std::clamp(fx1, 0.f, (float)(w - 1));
        if (dx0 != 0.f) fy0 += (nx0 - fx0) * dy0 / dx0;
        fx0 = nx0;
        if (dx1 != 0.f) fy1 += (nx1 - fx1) * dy1 / dx1;
        fx1 = nx1;

        const float ny0 = std::clamp(fy0, 0.f, (float)(h - 1));
        const float ny1 = std::clamp(fy1, 0.f, (float)(h - 1));
        if (dy0 != 0.f) fx0 += (ny0 - fy0) * dx0 / dy0;
        fy0 = ny0;
        if (dy1 != 0.f) fx1 += (ny1 - fy1) * dx1 / dy1;
        fy1 = ny1;

        fx0 = std::clamp(fx0, 0.f, (float)(w - 1));
        fx1 = std::clamp(fx1, 0.f, (float)(w - 1));
        fy0 = std::clamp(fy0, 0.f, (float)(h - 1));
        fy1 = std::clamp(fy1, 0.f, (float)(h - 1));

        int bx0 = (int)fx0, by0 = (int)fy0;
        const int bx1 = (int)fx1, by1 = (int)fy1;

        // Walk boundary counter-clockwise from (bx0,by0) to (bx1,by1)
        // collecting ray endpoints at each boundary pixel
        std::vector<std::pair<int, int>> boundary;  // (row, col)
        while (bx0 != bx1 || by0 != by1) {
            boundary.push_back({std::clamp(by0, 0, h - 1),
                                 std::clamp(bx0, 0, w - 1)});
            if      (bx0 == 0     && by0 < h - 1) by0++;
            else if (by0 == h - 1 && bx0 < w - 1) bx0++;
            else if (bx0 == w - 1 && by0 > 0)      by0--;
            else if (by0 == 0     && bx0 > 0)      bx0--;
            else break;
        }

        merged.at<uint8_t>(cam.i, cam.j) = 255;
        const cv::Point cam_pt(cam.j, cam.i);

        for (auto [end_row, end_col] : boundary) {
            uint8_t state = 255;
            cv::LineIterator it(merged, cam_pt, cv::Point(end_col, end_row));
            for (int p = 0; p < it.count; p++, ++it) {
                uint8_t& px = merged.at<uint8_t>(it.pos());
                if      (px == 0)   state = 0;
                else if (px == 255) state = 255;
                else                px = state;  // 127 → propagated state
            }
        }
    }

    return merged;
}

}  // namespace ransac
