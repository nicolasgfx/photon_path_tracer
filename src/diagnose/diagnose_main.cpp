// ─────────────────────────────────────────────────────────────────────
// diagnose_main.cpp – ppt_diagnose entry point
//
// Usage: ppt_diagnose <render.exr|render.png> [render_log.json]
//
// Loads a rendered EXR or PNG image, runs image oracle + diagnostics,
// prints quality report as JSON to stdout.
// Exit code 0 on success, 1 on error.  No GPU required.
// ─────────────────────────────────────────────────────────────────────
#include "diagnose/image_oracle.h"
#include "diagnose/image_verdict.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static bool load_exr(const std::string& path, std::vector<float>& pixels,
                     int& width, int& height) {
    float* data = nullptr;
    const char* err = nullptr;
    int ret = LoadEXR(&data, &width, &height, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            std::fprintf(stderr, "[Error] EXR load failed: %s\n", err);
            FreeEXRErrorMessage(err);
        }
        return false;
    }
    // Convert RGBA to RGB
    pixels.resize(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        pixels[i * 3 + 0] = data[i * 4 + 0];
        pixels[i * 3 + 1] = data[i * 4 + 1];
        pixels[i * 3 + 2] = data[i * 4 + 2];
    }
    free(data);
    return true;
}

static bool load_png(const std::string& path, std::vector<float>& pixels,
                     int& width, int& height) {
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::fprintf(stderr, "[Error] PNG load failed: %s\n", stbi_failure_reason());
        return false;
    }
    pixels.resize(width * height * 3);
    for (int i = 0; i < width * height * 3; ++i)
        pixels[i] = data[i] / 255.f;
    stbi_image_free(data);
    return true;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    std::string tail = s.substr(s.size() - suffix.size());
    std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);
    return tail == suffix;
}

static bool load_image(const std::string& path, std::vector<float>& pixels,
                       int& width, int& height) {
    if (ends_with(path, ".exr"))
        return load_exr(path, pixels, width, height);
    return load_png(path, pixels, width, height);
}

static void print_verdict_json(const ImageVerdict& v) {
    std::printf("{\n");
    std::printf("  \"quality\": {\n");
    std::printf("    \"noise_level\": %.4f,\n", v.noise_level);
    std::printf("    \"convergence_rate\": %.4f,\n", v.convergence_rate);
    std::printf("    \"converging_normally\": %s,\n",
                v.converging_normally ? "true" : "false");
    if (v.psnr_vs_reference >= 0.f)
        std::printf("    \"psnr\": %.2f,\n", v.psnr_vs_reference);
    if (v.ssim_vs_reference >= 0.f)
        std::printf("    \"ssim\": %.4f,\n", v.ssim_vs_reference);
    std::printf("    \"summary\": \"%s\"\n", v.summary.c_str());
    std::printf("  },\n");

    std::printf("  \"artifacts\": [\n");
    for (size_t i = 0; i < v.artifacts.size(); ++i) {
        const auto& a = v.artifacts[i];
        std::printf("    {\"type\": \"%s\", \"severity\": %.3f, \"description\": \"%s\"}%s\n",
                    artifact_type_name(a.type), a.severity,
                    a.description.c_str(),
                    (i + 1 < v.artifacts.size()) ? "," : "");
    }
    std::printf("  ],\n");

    std::printf("  \"corrections\": [\n");
    for (size_t i = 0; i < v.corrections.size(); ++i) {
        const auto& c = v.corrections[i];
        std::printf("    {\"target\": \"%s\", \"parameter\": \"%s\", "
                    "\"current\": %.4f, \"recommended\": %.4f, "
                    "\"rationale\": \"%s\"}%s\n",
                    c.target_skill.c_str(), c.parameter.c_str(),
                    c.current_value, c.recommended_value,
                    c.rationale.c_str(),
                    (i + 1 < v.corrections.size()) ? "," : "");
    }
    std::printf("  ]\n");
    std::printf("}\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: ppt_diagnose <render.exr|render.png> [render_log.json]\n");
        return 1;
    }

    std::string img_path = argv[1];

    // ── Load image (EXR or PNG) ─────────────────────────────────────
    std::vector<float> pixels;
    int width = 0, height = 0;
    if (!load_image(img_path, pixels, width, height)) {
        std::fprintf(stderr, "[Error] Failed to load: %s\n", img_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "[ppt_diagnose] Loaded %s (%dx%d)\n",
                 img_path.c_str(), width, height);

    // ── Run oracle ──────────────────────────────────────────────────
    ImageOracle oracle;
    ImageVerdict verdict = oracle.analyze(pixels.data(), width, height);

    // ── Output ──────────────────────────────────────────────────────
    print_verdict_json(verdict);
    return 0;
}
