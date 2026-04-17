// ---------------------------------------------------------------------
// report_listener.h — GTest event listener that writes detailed reports
// ---------------------------------------------------------------------
// Captures per-test pass/fail, duration, failure messages, and stdout
// output. At the end of the run, writes three files into the report
// directory:
//
//   report.txt    Human-readable detailed report (per-test)
//   report.json   Machine-readable JSON for offline analysis
//   summary.txt   Quick pass/fail overview with timing
//
// Usage: Register from main() before RUN_ALL_TESTS():
//
//   auto& listeners = UnitTest::GetInstance()->listeners();
//   listeners.Append(new ReportListener("tests/runs/20260222_031500"));
//
// The listener also captures each test's stdout by temporarily
// redirecting std::cout to a stringstream.
// ---------------------------------------------------------------------
#pragma once

// Prevent Windows.h min/max macros from conflicting with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Per-test result record ───────────────────────────────────────────
struct TestRecord {
    std::string suite;
    std::string name;
    bool        passed   = false;
    bool        skipped  = false;
    double      duration_sec = 0.0;
    std::string failure_message;   // empty if passed
    std::string stdout_capture;    // stdout emitted during the test
};

// ── Per-suite aggregate ──────────────────────────────────────────────
struct SuiteRecord {
    std::string name;
    int  total   = 0;
    int  passed  = 0;
    int  failed  = 0;
    int  skipped = 0;
    double duration_sec = 0.0;
};

// ── The listener ─────────────────────────────────────────────────────
class ReportListener : public ::testing::EmptyTestEventListener {
public:
    explicit ReportListener(const std::string& report_dir)
        : report_dir_(report_dir) {
        fs::create_directories(report_dir_);
    }

    // -- Programme-level events ----------------------------------------
    void OnTestProgramStart(const ::testing::UnitTest&) override {
        program_start_ = clock::now();
    }

    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        program_end_ = clock::now();
        write_report_txt(unit_test);
        write_report_json(unit_test);
        write_summary_txt(unit_test);

        double total_sec = elapsed(program_start_, program_end_);
        std::cout << "\n[ReportListener] Reports written to: " << report_dir_
                  << "  (" << format_duration(total_sec) << " total)\n";
    }

    // -- Suite-level events --------------------------------------------
    void OnTestSuiteStart(const ::testing::TestSuite& suite) override {
        current_suite_ = suite.name();
        suite_start_ = clock::now();
    }

    void OnTestSuiteEnd(const ::testing::TestSuite& suite) override {
        SuiteRecord sr;
        sr.name = suite.name();
        sr.total   = suite.total_test_count();
        sr.passed  = suite.successful_test_count();
        sr.failed  = suite.failed_test_count();
        sr.skipped = suite.skipped_test_count();
        sr.duration_sec = elapsed(suite_start_, clock::now());
        suites_.push_back(sr);
    }

    // -- Test-level events ---------------------------------------------
    void OnTestStart(const ::testing::TestInfo& /*info*/) override {
        // Capture stdout during each test
        saved_cout_ = std::cout.rdbuf();
        capture_stream_.str("");
        capture_stream_.clear();
        std::cout.rdbuf(capture_stream_.rdbuf());

        test_start_ = clock::now();
    }

    void OnTestEnd(const ::testing::TestInfo& info) override {
        // Restore stdout
        std::cout.rdbuf(saved_cout_);

        TestRecord rec;
        rec.suite   = info.test_suite_name();
        rec.name    = info.name();
        rec.passed  = info.result()->Passed();
        rec.skipped = info.result()->Skipped();
        rec.duration_sec = elapsed(test_start_, clock::now());
        rec.stdout_capture = capture_stream_.str();

        // Replay captured output to console so the user still sees it
        if (!rec.stdout_capture.empty()) {
            std::cout << rec.stdout_capture;
        }

        // Collect failure messages
        const auto* result = info.result();
        for (int i = 0; i < result->total_part_count(); ++i) {
            const auto& part = result->GetTestPartResult(i);
            if (part.failed()) {
                if (!rec.failure_message.empty())
                    rec.failure_message += "\n";
                rec.failure_message += part.file_name()
                    ? std::string(part.file_name()) + ":"
                      + std::to_string(part.line_number()) + ": "
                    : std::string("(unknown): ");
                rec.failure_message += part.message();
            }
        }

        tests_.push_back(rec);
    }

private:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;

    std::string       report_dir_;
    std::vector<TestRecord> tests_;
    std::vector<SuiteRecord> suites_;

    time_point program_start_;
    time_point program_end_;
    time_point suite_start_;
    time_point test_start_;
    std::string current_suite_;

    std::stringstream  capture_stream_;
    std::streambuf*    saved_cout_ = nullptr;

    static double elapsed(time_point a, time_point b) {
        return std::chrono::duration<double>(b - a).count();
    }

    static std::string format_duration(double sec) {
        int h = (int)(sec / 3600);
        int m = (int)(std::fmod(sec, 3600.0) / 60);
        double s = std::fmod(sec, 60.0);
        std::ostringstream os;
        if (h > 0) os << h << "h ";
        if (h > 0 || m > 0) os << m << "m ";
        os << std::fixed << std::setprecision(1) << s << "s";
        return os.str();
    }

    static std::string escape_json(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 16);
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if ((unsigned char)c < 0x20)
                        ; // skip control chars
                    else
                        out += c;
            }
        }
        return out;
    }

    // Count tests from our own records (respects gtest_filter)
    int count_passed() const {
        int n = 0; for (const auto& t : tests_) if (t.passed) ++n; return n;
    }
    int count_failed() const {
        int n = 0; for (const auto& t : tests_) if (!t.passed && !t.skipped) ++n; return n;
    }
    int count_skipped() const {
        int n = 0; for (const auto& t : tests_) if (t.skipped) ++n; return n;
    }
    int count_total() const { return (int)tests_.size(); }

    // ── report.txt ───────────────────────────────────────────────────
    void write_report_txt(const ::testing::UnitTest& /*ut*/) {
        std::ofstream f(report_dir_ + "/report.txt");
        if (!f) return;

        double total = elapsed(program_start_, program_end_);

        f << "===================================================================\n";
        f << "  PHOTON PATH TRACER — DETAILED TEST REPORT\n";
        f << "  Total time: " << format_duration(total) << "\n";
        f << "  Tests: " << count_total()
          << "  Passed: " << count_passed()
          << "  Failed: " << count_failed()
          << "  Skipped: " << count_skipped() << "\n";
        f << "===================================================================\n\n";

        // ── Per-suite sections ────────────────────────────────────────
        std::string last_suite;
        for (const auto& t : tests_) {
            if (t.suite != last_suite) {
                last_suite = t.suite;
                f << "-------------------------------------------------------------------\n";
                f << "  Suite: " << t.suite << "\n";
                // Find suite aggregate
                for (const auto& sr : suites_) {
                    if (sr.name == t.suite) {
                        f << "  " << sr.passed << "/" << sr.total << " passed"
                          << "  (" << format_duration(sr.duration_sec) << ")\n";
                        break;
                    }
                }
                f << "-------------------------------------------------------------------\n";
            }

            // Status icon + test name + duration
            const char* icon = t.passed ? "PASS" : (t.skipped ? "SKIP" : "FAIL");
            f << "  [" << icon << "]  " << t.name
              << "  (" << std::fixed << std::setprecision(2)
              << t.duration_sec << "s)\n";

            // Failure details
            if (!t.failure_message.empty()) {
                f << "         FAILURE:\n";
                std::istringstream lines(t.failure_message);
                std::string line;
                while (std::getline(lines, line))
                    f << "           " << line << "\n";
            }

            // Stdout capture (truncated to keep report manageable)
            if (!t.stdout_capture.empty()) {
                f << "         --- stdout ---\n";
                std::istringstream lines(t.stdout_capture);
                std::string line;
                int count = 0;
                while (std::getline(lines, line)) {
                    f << "         | " << line << "\n";
                    if (++count >= 200) {
                        f << "         | ... (truncated, "
                          << t.stdout_capture.size() << " bytes total)\n";
                        break;
                    }
                }
            }
        }

        // ── Failed tests summary ─────────────────────────────────────
        int nfail = count_failed();
        if (nfail > 0) {
            f << "\n===================================================================\n";
            f << "  FAILED TESTS (" << nfail << ")\n";
            f << "===================================================================\n";
            for (const auto& t : tests_) {
                if (!t.passed && !t.skipped) {
                    f << "  " << t.suite << "." << t.name << "\n";
                    if (!t.failure_message.empty()) {
                        std::istringstream lines(t.failure_message);
                        std::string line;
                        while (std::getline(lines, line))
                            f << "    " << line << "\n";
                    }
                }
            }
        }

        // ── Slowest tests ────────────────────────────────────────────
        f << "\n===================================================================\n";
        f << "  SLOWEST TESTS (top 20)\n";
        f << "===================================================================\n";
        std::vector<const TestRecord*> sorted;
        for (const auto& t : tests_) sorted.push_back(&t);
        std::sort(sorted.begin(), sorted.end(),
                  [](auto* a, auto* b) { return a->duration_sec > b->duration_sec; });
        int show = (std::min)((int)sorted.size(), 20);
        for (int i = 0; i < show; ++i) {
            const auto* t = sorted[i];
            f << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << t->duration_sec << "s  " << t->suite << "." << t->name << "\n";
        }

        // ── Suite timing summary ─────────────────────────────────────
        f << "\n===================================================================\n";
        f << "  SUITE TIMING SUMMARY\n";
        f << "===================================================================\n";
        std::vector<const SuiteRecord*> sorted_suites;
        for (const auto& s : suites_) sorted_suites.push_back(&s);
        std::sort(sorted_suites.begin(), sorted_suites.end(),
                  [](auto* a, auto* b) { return a->duration_sec > b->duration_sec; });
        for (const auto* s : sorted_suites) {
            f << "  " << std::setw(10) << std::fixed << std::setprecision(2)
              << s->duration_sec << "s  " << s->name
              << "  (" << s->passed << "/" << s->total << " passed)\n";
        }

        f << "\n===================================================================\n";
        f << "  END OF REPORT\n";
        f << "===================================================================\n";
    }

    // ── report.json ──────────────────────────────────────────────────
    void write_report_json(const ::testing::UnitTest& /*ut*/) {
        std::ofstream f(report_dir_ + "/report.json");
        if (!f) return;

        double total = elapsed(program_start_, program_end_);

        f << "{\n";
        f << "  \"total_time_sec\": " << std::fixed << std::setprecision(3) << total << ",\n";
        f << "  \"total_tests\": " << count_total() << ",\n";
        f << "  \"passed\": " << count_passed() << ",\n";
        f << "  \"failed\": " << count_failed() << ",\n";
        f << "  \"skipped\": " << count_skipped() << ",\n";

        // Suites
        f << "  \"suites\": [\n";
        for (size_t i = 0; i < suites_.size(); ++i) {
            const auto& s = suites_[i];
            f << "    {\"name\": \"" << escape_json(s.name) << "\""
              << ", \"total\": " << s.total
              << ", \"passed\": " << s.passed
              << ", \"failed\": " << s.failed
              << ", \"skipped\": " << s.skipped
              << ", \"duration_sec\": " << std::fixed << std::setprecision(3) << s.duration_sec
              << "}" << (i + 1 < suites_.size() ? "," : "") << "\n";
        }
        f << "  ],\n";

        // Tests
        f << "  \"tests\": [\n";
        for (size_t i = 0; i < tests_.size(); ++i) {
            const auto& t = tests_[i];
            f << "    {\"suite\": \"" << escape_json(t.suite) << "\""
              << ", \"name\": \"" << escape_json(t.name) << "\""
              << ", \"passed\": " << (t.passed ? "true" : "false")
              << ", \"skipped\": " << (t.skipped ? "true" : "false")
              << ", \"duration_sec\": " << std::fixed << std::setprecision(3) << t.duration_sec;

            if (!t.failure_message.empty())
                f << ", \"failure\": \"" << escape_json(t.failure_message) << "\"";

            if (!t.stdout_capture.empty()) {
                // Truncate large stdout in JSON to keep file size sane
                std::string cap = t.stdout_capture;
                if (cap.size() > 8192) {
                    cap = cap.substr(0, 8192) + "... (truncated)";
                }
                f << ", \"stdout\": \"" << escape_json(cap) << "\"";
            }

            f << "}" << (i + 1 < tests_.size() ? "," : "") << "\n";
        }
        f << "  ]\n";
        f << "}\n";
    }

    // ── summary.txt ──────────────────────────────────────────────────
    void write_summary_txt(const ::testing::UnitTest& /*ut*/) {
        std::ofstream f(report_dir_ + "/summary.txt");
        if (!f) return;

        double total = elapsed(program_start_, program_end_);

        f << "===================================================================\n";
        f << "  PHOTON PATH TRACER — OVERNIGHT TEST SUMMARY\n";
        f << "  Total time: " << format_duration(total) << "\n";
        f << "===================================================================\n\n";

        if (count_failed() == 0) {
            f << "  RESULT:  ALL " << count_total() << " TESTS PASSED\n\n";
        } else {
            f << "  RESULT:  " << count_failed() << " / "
              << count_total() << " TESTS FAILED\n\n";
        }

        f << "  Passed:  " << count_passed() << "\n";
        f << "  Failed:  " << count_failed() << "\n";
        f << "  Skipped: " << count_skipped() << "\n";
        f << "  Time:    " << format_duration(total) << "\n\n";

        // Failed test list
        if (count_failed() > 0) {
            f << "  FAILED:\n";
            for (const auto& t : tests_) {
                if (!t.passed && !t.skipped) {
                    f << "    - " << t.suite << "." << t.name
                      << "  (" << std::fixed << std::setprecision(2)
                      << t.duration_sec << "s)\n";
                }
            }
            f << "\n";
        }

        // Suite overview (sorted by duration)
        f << "  SUITE BREAKDOWN:\n";
        std::vector<const SuiteRecord*> sorted;
        for (const auto& s : suites_) sorted.push_back(&s);
        std::sort(sorted.begin(), sorted.end(),
                  [](auto* a, auto* b) { return a->duration_sec > b->duration_sec; });
        for (const auto* s : sorted) {
            const char* status = (s->failed > 0) ? "FAIL" : "OK  ";
            f << "    [" << status << "]  " << std::setw(10) << std::fixed
              << std::setprecision(1) << s->duration_sec << "s  " << s->name
              << "  (" << s->passed << "/" << s->total << ")\n";
        }

        // Diagnostic output extraction (key metrics from known test suites)
        f << "\n  KEY METRICS (from test stdout):\n";
        for (const auto& t : tests_) {
            if (t.stdout_capture.empty()) continue;
            // Extract lines with known metric tags
            std::istringstream lines(t.stdout_capture);
            std::string line;
            while (std::getline(lines, line)) {
                // Look for tagged metric lines
                if (line.find("[NEE-DirectOnly]") != std::string::npos ||
                    line.find("[PhotonIndirect]") != std::string::npos ||
                    line.find("[Combined") != std::string::npos ||
                    line.find("[Global]") != std::string::npos ||
                    line.find("[ShadowIndirect]") != std::string::npos ||
                    line.find("[Spatial]") != std::string::npos ||
                    line.find("[Decomposition]") != std::string::npos ||
                    line.find("[PhotonLobe]") != std::string::npos ||
                    line.find("[SpectralBin]") != std::string::npos ||
                    line.find("[CenterPixel]") != std::string::npos ||
                    line.find("[FullImage]") != std::string::npos ||
                    line.find("[Variance]") != std::string::npos ||
                    line.find("[DirectLighting]") != std::string::npos ||
                    line.find("[Energy]") != std::string::npos ||
                    line.find("Combined relErr:") != std::string::npos ||
                    line.find("[ShadowEdge]") != std::string::npos ||
                    line.find("Dense grid") != std::string::npos ||
                    line.find("DG/CPU") != std::string::npos ||
                    line.find("WARNING") != std::string::npos ||
                    line.find("PSNR") != std::string::npos ||
                    line.find("RMSE") != std::string::npos) {
                    // Trim leading whitespace
                    size_t start = line.find_first_not_of(" \t");
                    if (start != std::string::npos)
                        f << "    " << line.substr(start) << "\n";
                }
            }
        }

        // Image files
        f << "\n  OUTPUT FILES:\n";
        for (const auto& entry : fs::directory_iterator(report_dir_)) {
            if (entry.is_regular_file()) {
                auto sz = entry.file_size();
                std::string unit = "B";
                double size = (double)sz;
                if (sz > 1024*1024) { size /= 1024*1024; unit = "MB"; }
                else if (sz > 1024) { size /= 1024; unit = "KB"; }
                f << "    " << entry.path().filename().string()
                  << "  (" << std::fixed << std::setprecision(1) << size << " " << unit << ")\n";
            }
        }

        f << "\n===================================================================\n";
        f << "  Full report: report.txt\n";
        f << "  JSON data:   report.json\n";
        f << "===================================================================\n";
    }
};
