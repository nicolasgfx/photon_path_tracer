#pragma once
// ─────────────────────────────────────────────────────────────────────
// scene_profile_applicator.h – Map SceneProfile → RenderConfig
//
// Translates scene analysis insights into renderer parameters.
// Override cascade: CLI args > JSON config > SceneProfile > defaults
// ─────────────────────────────────────────────────────────────────────
#include "core/scene_profile.h"
#include "app/render_config.h"
#include "analyze/prepass_metrics.h"

// Apply scene-derived recommendations to a RenderConfig.
// Only writes fields that were left at their default values
// (i.e. not overridden by CLI or JSON config).
inline void apply_scene_profile(const SceneProfile& sp, RenderConfig& cfg) {
    // ── Core rendering ──────────────────────────────────────────────
    if (cfg.max_bounces == DEFAULT_MAX_BOUNCES_CAMERA)
        cfg.max_bounces = sp.recommended_max_bounces;

    // ── Firefly filter: relax threshold for glass/caustic scenes ────
    if (sp.has_glass && cfg.postfx.firefly_threshold <= FIREFLY_FILTER_THRESHOLD)
        cfg.postfx.firefly_threshold = 6.0f;

    // ── Caustic light tracing ────────────────────────────────────────
    // Apply coupling-driven budget when caustics are already on but
    // no explicit budget was set (i.e. still at zero / default).
    if (cfg.caustic_enabled && cfg.caustic_photons_per_frame == 0 &&
        sp.recommended_caustic_photons_per_frame > 0)
        cfg.caustic_photons_per_frame = sp.recommended_caustic_photons_per_frame;

    // Auto-enable only when user hasn't explicitly set via CLI/JSON.
    if (!cfg.caustic_enabled && cfg.caustic_photons_per_frame == 0) {
        cfg.caustic_enabled = sp.recommended_caustic_enabled;
        cfg.caustic_photons_per_frame = sp.recommended_caustic_photons_per_frame;
    }

    // Splat luminance: override only if still at default
    if (sp.recommended_caustic_max_splat_luminance > 0.f &&
        cfg.caustic_max_splat_luminance == DEFAULT_CAUSTIC_MAX_SPLAT_LUMINANCE)
        cfg.caustic_max_splat_luminance = sp.recommended_caustic_max_splat_luminance;

    // ── Light tree ──────────────────────────────────────────────────────
    if (cfg.light_tree_enabled == DEFAULT_LIGHT_TREE_ENABLED)
        cfg.light_tree_enabled = sp.recommended_light_tree_enabled;
}

// Refine SceneProfile using view-dependent pre-pass metrics.
// Call after run_prepass() and before re-applying to RenderConfig.
inline void refine_scene_profile(SceneProfile& sp, RenderConfig& cfg,
                                 const PrePassMetrics& m) {
    if (m.total_paths == 0) return;

    // ── High zero-path fraction → many paths contribute nothing ─────
    // Increase bounces.
    if (m.zero_path_fraction > 0.30f) {
        int new_bounces = (std::max)(sp.recommended_max_bounces, 12);
        if (new_bounces > sp.recommended_max_bounces) {
            std::printf("[PrePass] High zero-path fraction (%.1f%%) → bounces %d → %d\n",
                        m.zero_path_fraction * 100.f, sp.recommended_max_bounces, new_bounces);
            sp.recommended_max_bounces = new_bounces;
            cfg.max_bounces = new_bounces;
        }
    }

    // ── High average bounce depth → scene needs more bounces ────────
    if (m.avg_bounce_depth > 6.0f) {
        int needed = (int)(m.avg_bounce_depth * 1.5f);
        needed = (std::min)(needed, 20);
        if (needed > sp.recommended_max_bounces) {
            std::printf("[PrePass] High avg bounce (%.1f) → bounces %d → %d\n",
                        m.avg_bounce_depth, sp.recommended_max_bounces, needed);
            sp.recommended_max_bounces = needed;
            cfg.max_bounces = needed;
        }
    }

    // ── High variance P99 → tighten sample luminance clamp ──────────
    if (m.variance_p99 > 1.0f) {
        float new_clamp = (std::min)(cfg.max_sample_luminance,
                                     m.variance_p99 * 5.f);
        if (new_clamp < cfg.max_sample_luminance) {
            std::printf("[PrePass] High variance p99 (%.2f) → max_sample_lum %.0f → %.0f\n",
                        m.variance_p99, cfg.max_sample_luminance, new_clamp);
            cfg.max_sample_luminance = new_clamp;
        }
    }
}
