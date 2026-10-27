"""ParameterSpec registry: single source of truth binding pipeline config
fields to the Obsidian Studio inspector widgets.

Collapses the three config<->widget mappers
(``_on_preset_change``, ``_apply_diagnostics_to_ui``,
``_get_current_config``) into one declarative table, so a parameter is
declared once and applied everywhere (the D1/D5 refactor from the audit).

Execution-mode gating (which widgets are live in PureDSP vs Neural AI vs
Hybrid) is also declared here, so ``apply_mode_visibility`` is a single
loop instead of a hand-written per-widget block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, List, Optional


@dataclass(frozen=True)
class ParameterSpec:
    """Declares how one ``PipelineConfig`` field is bound to one widget.

    config_field -- ``PipelineConfig`` attribute name
    widget_attr  -- ``PureScaleApp`` attribute holding the live widget
    kind         -- "slider" (LabeledSlider), "switch" (CTkSwitch),
                    or "segment" (CTkSegmentedButton)
    default      -- value used when a source omits the field
    enabled_modes -- set of mode labels ("PureDSP"/"Neural AI"/"Hybrid")
                    in which the widget is enabled; ``None`` means every mode.
    """
    config_field: str
    widget_attr: str
    kind: str
    default: Any
    enabled_modes: Optional[FrozenSet[str]] = None


@dataclass
class _WidgetBinding:
    """A registry spec wired to its live widget instance."""
    spec: ParameterSpec
    widget: Any
    get: Callable[[], Any]
    set: Callable[[Any], None]


# Order mirrors the sidebar layout so application is deterministic.
# Mode-gated widgets carry ``enabled_modes``; everything else is always live.
REGISTRY: List[ParameterSpec] = [
    # Performance & Telemetry
    ParameterSpec("auto_style", "auto_style_switch", "switch", False),
    ParameterSpec("enable_diagnostics", "diagnostics_switch", "switch", True),
    ParameterSpec("enable_dehaze", "dehaze_switch", "switch", False),
    ParameterSpec("dehaze_strength", "dehaze_slider", "slider", 0.50),
    ParameterSpec("enable_pyramid", "pyramid_switch", "switch", True),
    ParameterSpec("pyramid_micro_texture", "pyramid_detail_slider", "slider", 1.20),
    ParameterSpec("pyramid_structure_boost", "pyramid_struct_slider", "slider", 1.10),
    ParameterSpec("pyramid_dynamic_range", "pyramid_dynrange_slider", "slider", 0.0),
    ParameterSpec("enable_semantic_guidance", "semantic_switch", "switch", True),
    # Spatial Super-Resolution
    ParameterSpec("scale", "scale_slider", "slider", 2.0),
    # Detail Clarity (disabled in Neural AI)
    ParameterSpec("sharpen_strength", "cas_slider", "slider", 1.10,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    # Structural Denoise (disabled in Neural AI)
    ParameterSpec("enable_denoise", "denoise_switch", "switch", True,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("denoise_intensity", "denoise_slider", "slider", 40,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    # Dynamic Range & Color (disabled in Neural AI)
    ParameterSpec("enable_contrast", "contrast_switch", "switch", True,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("contrast_boost", "contrast_slider", "slider", 1.80,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("brightness_shift", "bright_slider", "slider", 0,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("enable_local_tone", "local_tone_switch", "switch", False,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("local_tone_strength", "tone_strength_slider", "slider", 0.50,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("highlight_recovery", "hl_recovery_slider", "slider", 0.50,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("shadow_boost", "shadow_boost_slider", "slider", 0.50,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("vibrance_boost", "vibrance_slider", "slider", 1.10,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("color_temperature", "temp_slider", "slider", 0,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    ParameterSpec("color_tint", "tint_slider", "slider", 0,
                  enabled_modes=frozenset({"PureDSP", "Hybrid"})),
    # Pre-Restoration Conditioning
    ParameterSpec("depixel_strength", "depixel_slider", "slider", 0),
    ParameterSpec("deblur_strength", "deblur_slider", "slider", 0),
    # Facial Retouching
    ParameterSpec("portrait_smooth", "smooth_slider", "slider", 35),
    ParameterSpec("eye_clarity", "eye_slider", "slider", 1.30),
    # Neural AI Execution (disabled in PureDSP)
    ParameterSpec("enable_fast_2x", "fast_2x_switch", "switch", False,
                  enabled_modes=frozenset({"Neural AI", "Hybrid"})),
    # Memory & Output
    ParameterSpec("max_megapixels", "max_mp_slider", "slider", 40.0),
]

# Segmented buttons handled outside the registry because their values need
# an enum/lowercase transform (mode, sr_style, output_format).
SEGMENT_WIDGET_ATTRS = ("mode_seg", "style_seg", "fmt_seg")


def _slider_get(widget) -> Callable[[], float]:
    return lambda: widget.get()


def _slider_set(widget) -> Callable[[float], None]:
    return lambda value: widget.set(value)


def _switch_get(widget) -> Callable[[], bool]:
    return lambda: bool(widget.get())


def _switch_set(widget) -> Callable[[bool], None]:
    return lambda value: widget.select() if value else widget.deselect()


def _segment_get(widget) -> Callable[[], str]:
    return lambda: widget.get()


def _segment_set(widget) -> Callable[[str], None]:
    return lambda value: widget.set(value)


def _make_getter(widget: Any, kind: str) -> Callable[[], Any]:
    if kind == "slider":
        return _slider_get(widget)
    if kind == "switch":
        return _switch_get(widget)
    if kind == "segment":
        return _segment_get(widget)
    raise ValueError(f"unknown parameter kind {kind!r}")


def _make_setter(widget: Any, kind: str) -> Callable[[Any], None]:
    if kind == "slider":
        return _slider_set(widget)
    if kind == "switch":
        return _switch_set(widget)
    if kind == "segment":
        return _segment_set(widget)
    raise ValueError(f"unknown parameter kind {kind!r}")


def build_param_bindings(app: Any) -> List[_WidgetBinding]:
    """Pairs each registry spec with its live widget and get/set callables.

    Called once after ``_populate_sidebar`` has created every widget.
    """
    bindings: List[_WidgetBinding] = []
    for spec in REGISTRY:
        widget = getattr(app, spec.widget_attr, None)
        if widget is None:
            continue
        bindings.append(
            _WidgetBinding(
                spec=spec,
                widget=widget,
                get=_make_getter(widget, spec.kind),
                set=_make_setter(widget, spec.kind),
            )
        )
    return bindings


def param_widget_attrs() -> List[str]:
    """Every widget attribute driven by the registry (plus the special
    segments), used to enable/disable the whole inspector during a job."""
    return [spec.widget_attr for spec in REGISTRY] + list(SEGMENT_WIDGET_ATTRS)
