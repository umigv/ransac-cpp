#include "ransac/plane.hpp"
#include "ransac/occu.hpp"

#include <BS_thread_pool.hpp>
#include <H5Cpp.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ── HDF5 helpers ─────────────────────────────────────────────────────────────

static cv::Mat read_depth_frame(H5::H5File& file, const std::string& name, int frame) {
    H5::DataSet   ds     = file.openDataSet(name);
    H5::DataSpace fspace = ds.getSpace();
    std::vector<hsize_t> dims(fspace.getSimpleExtentNdims());
    fspace.getSimpleExtentDims(dims.data());

    hsize_t offset[3] = {(hsize_t)frame, 0, 0};
    hsize_t count[3]  = {1, dims[1], dims[2]};
    fspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    hsize_t memdims[2] = {dims[1], dims[2]};
    H5::DataSpace mspace(2, memdims);

    cv::Mat mat((int)dims[1], (int)dims[2], CV_32F);
    ds.read(mat.data, H5::PredType::NATIVE_FLOAT, mspace, fspace);
    return mat;
}

static cv::Mat read_image_frame(H5::H5File& file, const std::string& name, int frame) {
    H5::DataSet   ds     = file.openDataSet(name);
    H5::DataSpace fspace = ds.getSpace();
    std::vector<hsize_t> dims(fspace.getSimpleExtentNdims());
    fspace.getSimpleExtentDims(dims.data());

    hsize_t offset[4] = {(hsize_t)frame, 0, 0, 0};
    hsize_t count[4]  = {1, dims[1], dims[2], dims[3]};
    fspace.selectHyperslab(H5S_SELECT_SET, count, offset);

    hsize_t memdims[3] = {dims[1], dims[2], dims[3]};
    H5::DataSpace mspace(3, memdims);

    cv::Mat mat((int)dims[1], (int)dims[2], CV_8UC3);
    ds.read(mat.data, H5::PredType::NATIVE_UINT8, mspace, fspace);
    return mat;
}

// ── Per-camera pipeline ───────────────────────────────────────────────────────

struct CameraResult {
    cv::Mat         full_occ;
    Eigen::Vector3f coeffs;
    float           angle_deg;
};

static CameraResult process_camera(
    const cv::Mat&                  image,
    const cv::Mat&                  raw_depths,
    const ransac::Intrinsics&       intr,
    const ransac::GridConfiguration& drive_conf,
    const ransac::GridConfiguration& block_conf,
    int iters, int kh, int kw, float tol,
    std::mt19937 rng)           // passed by value — each camera owns its RNG
{
    cv::Mat depths = raw_depths.clone();
    ransac::clean_depths(depths);

    auto [driveable, coeffs] = ransac::hsv_and_ransac(
        image, depths, iters, kh, kw, tol, rng);

    auto real_c = ransac::real_coeffs(coeffs, intr);
    float angle = ransac::real_angle(real_c);

    cv::Mat not_driveable = (driveable == 0);
    auto drive_ppc = ransac::create_point_cloud(driveable,     depths);
    auto drive_rpc = ransac::pixel_to_real(drive_ppc, real_c, intr, (float)M_PI_4);
    auto block_ppc = ransac::create_point_cloud(not_driveable, depths);
    auto block_rpc = ransac::pixel_to_real(block_ppc, real_c, intr, (float)M_PI_4);

    auto drive_occ = ransac::occupancy_grid(drive_rpc, drive_conf);
    auto block_occ = ransac::occupancy_grid(block_rpc, block_conf);

    return {ransac::composite(drive_occ, block_occ), coeffs, angle * 180.f / (float)M_PI};
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.hdf5> [frame]\n";
        return 1;
    }

    H5::H5File file(argv[1], H5F_ACC_RDONLY);

    H5::DataSpace sp = file.openDataSet("depth_maps").getSpace();
    std::vector<hsize_t> dims(sp.getSimpleExtentNdims());
    sp.getSimpleExtentDims(dims.data());
    const int num_frames = (int)dims[0];

    std::mt19937 main_rng(std::random_device{}());

    int frame = (argc >= 3) ? std::stoi(argv[2]) : -1;
    if (frame < 0) {
        frame = std::uniform_int_distribution<int>(1, num_frames - 3)(main_rng);
        std::cout << "Using randomised frame: " << frame << "\n";
    }
    frame = std::min(frame, num_frames - 2);  // need frame+1 for cam 1

    // ── Load two frames in main thread (HDF5 is not thread-safe) ─────────────
    // Cam 0 = frame N,  Cam 1 = frame N+1 (simulates two physical cameras)
    auto load = [&](int f) -> std::pair<cv::Mat, cv::Mat> {
        cv::Mat full = read_image_frame(file, "images", f);
        cv::Mat img  = full(cv::Rect(0, 0, full.cols / 2, full.rows)).clone();
        cv::Mat dep  = read_depth_frame(file, "depth_maps", f);
        return {img, dep};
    };
    auto [img0, dep0] = load(frame);
    auto [img1, dep1] = load(frame + 1);

    const int   W   = dep0.cols, H = dep0.rows;
    const float fx  = 360.f;
    const ransac::Intrinsics       intr      {(float)W/2, (float)H/2, fx, fx};
    const ransac::GridConfiguration drive_conf{5000, 5000, 50, 2};
    const ransac::GridConfiguration block_conf{5000, 5000, 50, 1};
    const int   iters = 50, kh = 1, kw = 16;
    const float tol   = 0.1f;

    // Independent RNGs — re-seeded identically before each benchmark mode
    // so sequential and parallel explore the same RANSAC sample space.
    const uint32_t seed0 = main_rng(), seed1 = main_rng();
    std::mt19937 rng0(seed0), rng1(seed1);

    BS::thread_pool pool(2);
    constexpr int N_ITERS = 10;  // timed repetitions per mode

    auto median_ms = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // ── Warmup: one pass of each mode, results discarded ─────────────────────
    // This brings img/dep data into cache equally for both modes.
    std::cout << "Warming up..." << std::flush;
    rng0.seed(seed0); rng1.seed(seed1);
    process_camera(img0, dep0, intr, drive_conf, block_conf, iters, kh, kw, tol, rng0);
    process_camera(img1, dep1, intr, drive_conf, block_conf, iters, kh, kw, tol, rng1);

    rng0.seed(seed0); rng1.seed(seed1);
    { auto f0 = pool.submit_task([&]{ return process_camera(img0, dep0, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed0)); });
      auto f1 = pool.submit_task([&]{ return process_camera(img1, dep1, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed1)); });
      f0.get(); f1.get(); }
    std::cout << " done\n";

    // ── Sequential (N_ITERS repetitions) ─────────────────────────────────────
    std::vector<double> seq_times;
    CameraResult seq0_result, seq1_result;
    for (int i = 0; i < N_ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        seq0_result = process_camera(img0, dep0, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed0));
        seq1_result = process_camera(img1, dep1, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed1));
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // ── Parallel (N_ITERS repetitions) ───────────────────────────────────────
    std::vector<double> par_times;
    CameraResult par0_result, par1_result;
    for (int i = 0; i < N_ITERS; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto f0 = pool.submit_task([&]{ return process_camera(img0, dep0, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed0)); });
        auto f1 = pool.submit_task([&]{ return process_camera(img1, dep1, intr, drive_conf, block_conf, iters, kh, kw, tol, std::mt19937(seed1)); });
        par0_result = f0.get();
        par1_result = f1.get();
        auto t1 = std::chrono::high_resolution_clock::now();
        par_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double ms_seq = median_ms(seq_times);
    double ms_par = median_ms(par_times);

    // ── Merge occupancy grids (same as live_full_2.py np.maximum) ─────────────
    cv::Mat merged;
    cv::max(par0_result.full_occ, par1_result.full_occ, merged);

    const int occ_h = merged.rows, occ_w = merged.cols;
    ransac::VirtualCamera vcam{occ_h - 1, occ_w / 2,
                                (float)(3.0 * M_PI / 4.0),
                                (float)(M_PI / 2.0)};
    cv::Mat los = ransac::create_los_grid(merged.clone(), {vcam});

    // ── Results ───────────────────────────────────────────────────────────────
    std::cout << "\n=== Camera 0 (frame " << frame   << ") ===\n"
              << "  coeffs: [" << par0_result.coeffs(0) << ", " << par0_result.coeffs(1) << ", " << par0_result.coeffs(2) << "]\n"
              << "  angle:  " << par0_result.angle_deg << " deg\n";
    std::cout << "\n=== Camera 1 (frame " << frame+1 << ") ===\n"
              << "  coeffs: [" << par1_result.coeffs(0) << ", " << par1_result.coeffs(1) << ", " << par1_result.coeffs(2) << "]\n"
              << "  angle:  " << par1_result.angle_deg << " deg\n";
    std::cout << "\n=== Timing (median of " << N_ITERS << " runs, cache-warm) ===\n"
              << "  sequential: " << ms_seq << " ms\n"
              << "  parallel:   " << ms_par << " ms\n"
              << "  speedup:    " << ms_seq / ms_par << "x\n" << std::flush;

    // ── Display ───────────────────────────────────────────────────────────────
    const char* disp = std::getenv("DISPLAY");
    if (disp && disp[0] != '\0') {
        const cv::Size sz(600, 600);
        auto show = [&](const std::string& name, cv::Mat m) {
            cv::Mat tmp; cv::resize(m, tmp, sz, 0, 0, cv::INTER_NEAREST);
            cv::imshow(name, tmp);
        };

        cv::Mat occ0_bgr, occ1_bgr, merged_bgr, los_bgr;
        cv::cvtColor(par0_result.full_occ, occ0_bgr,  cv::COLOR_GRAY2BGR);
        cv::cvtColor(par1_result.full_occ, occ1_bgr,  cv::COLOR_GRAY2BGR);
        cv::cvtColor(merged,        merged_bgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(los,           los_bgr,    cv::COLOR_GRAY2BGR);

        show("cam 0 occ",    occ0_bgr);
        show("cam 1 occ",    occ1_bgr);
        show("merged area",  merged_bgr);
        show("line of sight", los_bgr);
        cv::waitKey(0);
    }

    return 0;
}
