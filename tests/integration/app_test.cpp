// ─────────────────────────────────────────────────────────────────────
// app_test.cpp – Phase 13 test: Application Layer
//
// Tests:
//   1. RenderConfig defaults match config.h constants
//   2. FrameBuffer resize / clear / tonemap
//   3. FrameBuffer get_pixel averaging
//   4. CLI args: empty → defaults
//   5. CLI args: --spp 64 --width 512
//   6. CLI args: --mode direct
//   7. CLI args: --headless flag
//   8. CLI args: positional scene file
//   9. Font overlay: stamp text into buffer
//  10. Font overlay: measureTextWidth
//  11. RenderLog JSON writer
//  12. Debug state defaults
//
// Pure CPU — no GPU required.
// ─────────────────────────────────────────────────────────────────────
#include "app/render_config.h"
#include "app/cli_args.h"
#include "app/viewer.h"
#include "debug/font_overlay.h"
#include "debug/render_log.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

// ── Test helpers ─────────────────────────────────────────────────────
static int g_tests_run    = 0;
static int g_tests_passed = 0;

#define TEST_BEGIN(name)                                 \
    do {                                                 \
        g_tests_run++;                                   \
        printf("\n[TEST %d] %s\n", g_tests_run, name);  \
    } while(0)

#define TEST_PASS(name)                                  \
    do {                                                 \
        g_tests_passed++;                                \
        printf("[PASS] %s\n", name);                     \
    } while(0)

#define EXPECT_TRUE(cond, msg)                           \
    do {                                                 \
        if (!(cond)) {                                   \
            printf("[FAIL] %s: %s\n", msg, #cond);       \
            return;                                      \
        }                                                \
    } while(0)

// ─────────────────────────────────────────────────────────────────────
// Test 1: RenderConfig defaults
// ─────────────────────────────────────────────────────────────────────
static void test_render_config_defaults() {
    TEST_BEGIN("RenderConfig defaults");

    RenderConfig cfg;
    EXPECT_TRUE(cfg.image_width == DEFAULT_IMAGE_WIDTH, "width");
    EXPECT_TRUE(cfg.image_height == DEFAULT_IMAGE_HEIGHT, "height");
    EXPECT_TRUE(cfg.samples_per_pixel == DEFAULT_SPP, "spp");
    EXPECT_TRUE(cfg.max_bounces == DEFAULT_MAX_BOUNCES_CAMERA, "bounces");
    EXPECT_TRUE(cfg.denoiser_enabled == DEFAULT_DENOISER_ENABLED, "denoiser");
    EXPECT_TRUE(cfg.mode == RenderMode::Combined, "mode");

    TEST_PASS("RenderConfig defaults");
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: FrameBuffer resize / clear / tonemap
// ─────────────────────────────────────────────────────────────────────
static void test_framebuffer_ops() {
    TEST_BEGIN("FrameBuffer resize/clear/tonemap");

    FrameBuffer fb;
    fb.resize(16, 16);

    EXPECT_TRUE(fb.width == 16, "width 16");
    EXPECT_TRUE(fb.height == 16, "height 16");
    EXPECT_TRUE(fb.rgb.size() == 16 * 16 * 3, "rgb size");
    EXPECT_TRUE(fb.sample_count.size() == 16 * 16, "sample count size");
    EXPECT_TRUE(fb.srgb.size() == 16 * 16 * 4, "srgb size");

    // Fill one pixel with known value
    fb.rgb[0] = 1.0f; fb.rgb[1] = 0.5f; fb.rgb[2] = 0.0f;
    fb.sample_count[0] = 1.0f;

    fb.tonemap(1.0f);
    // After ACES tonemap, R=1.0 → some positive sRGB value
    EXPECT_TRUE(fb.srgb[0] > 0, "red channel > 0 after tonemap");
    EXPECT_TRUE(fb.srgb[3] == 255, "alpha = 255");

    fb.clear();
    EXPECT_TRUE(fb.rgb[0] == 0.f, "cleared to zero");
    EXPECT_TRUE(fb.sample_count[0] == 0.f, "sample count cleared");

    TEST_PASS("FrameBuffer resize/clear/tonemap");
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: FrameBuffer get_pixel averaging
// ─────────────────────────────────────────────────────────────────────
static void test_framebuffer_get_pixel() {
    TEST_BEGIN("FrameBuffer get_pixel");

    FrameBuffer fb;
    fb.resize(4, 4);

    // Accumulate 2 samples at pixel (1,1)
    int idx = (1 * 4 + 1) * 3;
    fb.rgb[idx + 0] = 2.0f;  // total R
    fb.rgb[idx + 1] = 1.0f;  // total G
    fb.rgb[idx + 2] = 0.4f;  // total B
    fb.sample_count[1 * 4 + 1] = 2.0f;

    Color3 c = fb.get_pixel(1, 1);
    EXPECT_TRUE(std::abs(c.r - 1.0f) < 0.001f, "avg R = 1.0");
    EXPECT_TRUE(std::abs(c.g - 0.5f) < 0.001f, "avg G = 0.5");
    EXPECT_TRUE(std::abs(c.b - 0.2f) < 0.001f, "avg B = 0.2");

    // Zero samples → zero pixel
    Color3 z = fb.get_pixel(0, 0);
    EXPECT_TRUE(z.r == 0.f && z.g == 0.f && z.b == 0.f, "zero samples = zero");

    TEST_PASS("FrameBuffer get_pixel");
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: CLI args empty → defaults
// ─────────────────────────────────────────────────────────────────────
static void test_cli_empty() {
    TEST_BEGIN("CLI: empty → defaults");

    char* argv[] = { (char*)"test" };
    Options opt = parse_args(1, argv);

    EXPECT_TRUE(opt.scene_file.empty(), "no scene file");
    EXPECT_TRUE(opt.config.samples_per_pixel == DEFAULT_SPP, "default spp");
    EXPECT_TRUE(!opt.headless, "not headless");
    EXPECT_TRUE(!opt.help, "no help");

    TEST_PASS("CLI: empty → defaults");
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: CLI args --spp 64 --width 512
// ─────────────────────────────────────────────────────────────────────
static void test_cli_spp_width() {
    TEST_BEGIN("CLI: --spp 64 --width 512");

    char* argv[] = { (char*)"test", (char*)"--spp", (char*)"64",
                     (char*)"--width", (char*)"512" };
    Options opt = parse_args(5, argv);

    EXPECT_TRUE(opt.config.samples_per_pixel == 64, "spp=64");
    EXPECT_TRUE(opt.config.image_width == 512, "width=512");

    TEST_PASS("CLI: --spp 64 --width 512");
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: CLI args --mode direct
// ─────────────────────────────────────────────────────────────────────
static void test_cli_mode() {
    TEST_BEGIN("CLI: --mode direct");

    char* argv[] = { (char*)"test", (char*)"--mode", (char*)"direct" };
    Options opt = parse_args(3, argv);

    EXPECT_TRUE(opt.config.mode == RenderMode::DirectOnly, "direct mode");

    TEST_PASS("CLI: --mode direct");
}

// ─────────────────────────────────────────────────────────────────────
// Test 7: CLI args --headless
// ─────────────────────────────────────────────────────────────────────
static void test_cli_headless() {
    TEST_BEGIN("CLI: --headless");

    char* argv[] = { (char*)"test", (char*)"--headless" };
    Options opt = parse_args(2, argv);

    EXPECT_TRUE(opt.headless, "headless=true");

    TEST_PASS("CLI: --headless");
}

// ─────────────────────────────────────────────────────────────────────
// Test 8: CLI args positional scene file
// ─────────────────────────────────────────────────────────────────────
static void test_cli_positional() {
    TEST_BEGIN("CLI: positional scene file");

    char* argv[] = { (char*)"test", (char*)"scene.pbrt" };
    Options opt = parse_args(2, argv);

    EXPECT_TRUE(opt.scene_file == "scene.pbrt", "scene file parsed");

    TEST_PASS("CLI: positional scene file");
}

// ─────────────────────────────────────────────────────────────────────
// Test 9: Font overlay stamp text
// ─────────────────────────────────────────────────────────────────────
static void test_font_stamp() {
    TEST_BEGIN("Font overlay: stamp text");

    uint32_t W = 64, H = 16;
    std::vector<uint8_t> pixels(W * H * 4, 0);

    font_overlay::stampText(pixels, W, H, "Hi", 2, 2, 255, 255, 255, 1.0f, 1);

    // Check that some pixels were written (non-zero)
    bool found_nonzero = false;
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0) {
            found_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(found_nonzero, "text pixels should be non-zero");

    TEST_PASS("Font overlay: stamp text");
}

// ─────────────────────────────────────────────────────────────────────
// Test 10: Font overlay measureTextWidth
// ─────────────────────────────────────────────────────────────────────
static void test_font_measure() {
    TEST_BEGIN("Font overlay: measureTextWidth");

    int w1 = font_overlay::measureTextWidth("A", 1);
    int w5 = font_overlay::measureTextWidth("ABCDE", 1);

    EXPECT_TRUE(w1 == 6, "1 char = 6 pixels (5+1 spacing)");
    EXPECT_TRUE(w5 == 30, "5 chars = 30 pixels");

    int w2x = font_overlay::measureTextWidth("A", 2);
    EXPECT_TRUE(w2x == 12, "scale 2: 1 char = 12 pixels");

    TEST_PASS("Font overlay: measureTextWidth");
}

// ─────────────────────────────────────────────────────────────────────
// Test 11: RenderLog JSON writer
// ─────────────────────────────────────────────────────────────────────
static void test_render_log_json() {
    TEST_BEGIN("RenderLog JSON writer");

    RenderLog log;
    log.timestamp = "2025-01-01T00:00:00";
    log.scene_name = "test_scene";
    log.image_width = 800;
    log.image_height = 600;
    log.accumulated_spp = 256;
    log.max_bounces_camera = 8;

    std::string path = "test_render_log_tmp.json";
    bool ok = render_log_write_json(log, path);
    EXPECT_TRUE(ok, "write should succeed");

    // Read back and verify it contains expected content
    FILE* f = fopen(path.c_str(), "r");
    EXPECT_TRUE(f != nullptr, "file should exist");
    char buf[4096];
    size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[len] = '\0';

    EXPECT_TRUE(strstr(buf, "render_log_v5") != nullptr, "schema v5");
    EXPECT_TRUE(strstr(buf, "test_scene") != nullptr, "scene name present");
    EXPECT_TRUE(strstr(buf, "\"accumulated_spp\": 256") != nullptr, "spp present");

    // Clean up
    std::remove(path.c_str());

    TEST_PASS("RenderLog JSON writer");
}

// ─────────────────────────────────────────────────────────────────────
// Test 12: Debug state defaults
// ─────────────────────────────────────────────────────────────────────
static void test_debug_state() {
    TEST_BEGIN("DebugState defaults");

    DebugState ds;
    EXPECT_TRUE(!ds.show_normals, "normals off");
    EXPECT_TRUE(!ds.show_stats_overlay, "stats off");
    EXPECT_TRUE(!ds.show_noise_map, "noise map off");

    TEST_PASS("DebugState defaults");
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────
int main() {
    printf("=== Application Layer Tests (Phase 13) ===\n");

    test_render_config_defaults();
    test_framebuffer_ops();
    test_framebuffer_get_pixel();
    test_cli_empty();
    test_cli_spp_width();
    test_cli_mode();
    test_cli_headless();
    test_cli_positional();
    test_font_stamp();
    test_font_measure();
    test_render_log_json();
    test_debug_state();

    printf("\n========================================\n");
    printf(" Application Layer: %d / %d passed",
           g_tests_passed, g_tests_run);
    if (g_tests_run != g_tests_passed)
        printf(" (%d FAILED)", g_tests_run - g_tests_passed);
    printf("\n========================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
