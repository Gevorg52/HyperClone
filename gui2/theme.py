"""gui2 - one coherent Apex Reality dark theme (UI-5).

A single source of truth for the wizard's colors. Root cause of the old
"mixed black/white, black-text-on-black" render: two independent theme systems
disagreed. Gradio's Soft theme styled its own components via `.dark`-scoped
tokens that follow the browser's prefers-color-scheme, while `theme.css` styled
our custom HTML cards via `--apex-*` tokens that flipped under a *separate*
`@media (prefers-color-scheme: light)` rule. When the two disagreed, one surface
went light and the next stayed dark.

Fix: pin ONE dark palette for BOTH modes. Gradio emits base tokens under
`:root {}` and dark tokens under `:root.dark {}` (themes/base.py). Setting each
token AND its `_dark` twin to the same dark value makes the browser's color
preference irrelevant - every surface resolves to the same dark value whether or
not the `.dark` class is present. `theme.css` then only does layout; it no longer
fights the theme, and the `--apex-*` tokens mirror this palette with no media flip.
"""

from __future__ import annotations

import gradio as gr

# --------------------------------------------------------------------------
# The palette (single source; also mirrored in theme.css as --apex-* vars)
# --------------------------------------------------------------------------
ACCENT = "#7c6cff"          # single brand accent (borders / stars / links on dark)
ACCENT_2 = "#a99bff"        # lighter accent (hover / links)
ACCENT_SOFT = "rgba(124, 108, 255, 0.16)"
# Deeper accent for surfaces that carry WHITE text (primary buttons, step badges):
# white on #6656e6 is 5.2:1 (passes AA); white on #7c6cff is only 3.6:1 (fails).
ACCENT_BTN = "#6656e6"

APP_BG = "#0e0f13"          # deepest app background
PANEL = "#16181f"           # blocks / panels
CARD = "#1c1f28"            # elevated cards / inputs
CODE_BG = "#0b0c10"         # code / log
BORDER = "#2a2f3a"          # borders (a touch brighter than blocks for edges)
BORDER_SOFT = "#222633"

TEXT = "#e7e9ee"            # primary text (contrast >= 12:1 on APP_BG)
MUTED = "#a3aab8"           # secondary text (contrast >= 4.9:1 on PANEL)
WHITE = "#ffffff"

GOOD = "#3ecf8e"
WARN = "#f0a84a"
DANGER = "#ff6b6b"

# Token -> value. Every one gets a matching `_dark` twin below so both the
# `:root {}` and `:root.dark {}` blocks Gradio emits carry identical dark values.
_TOKENS: dict[str, str] = {
    # global surfaces + text
    "body_background_fill": APP_BG,
    "body_text_color": TEXT,
    "body_text_color_subdued": MUTED,
    "background_fill_primary": PANEL,
    "background_fill_secondary": CARD,
    "border_color_primary": BORDER,
    "border_color_accent": ACCENT,
    "border_color_accent_subdued": ACCENT_SOFT,
    "color_accent": ACCENT,
    "color_accent_soft": ACCENT_SOFT,
    "link_text_color": ACCENT_2,
    "link_text_color_hover": WHITE,
    "link_text_color_active": ACCENT_2,
    "link_text_color_visited": ACCENT_2,
    # blocks / panels
    "block_background_fill": PANEL,
    "block_border_color": BORDER,
    "block_label_background_fill": PANEL,
    "block_label_text_color": MUTED,
    "block_label_border_color": BORDER,
    "block_title_text_color": TEXT,
    "block_title_background_fill": "transparent",
    "block_info_text_color": MUTED,
    "panel_background_fill": PANEL,
    "panel_border_color": BORDER,
    "accordion_text_color": TEXT,
    "stat_background_fill": ACCENT,
    # inputs
    "input_background_fill": CARD,
    "input_background_fill_focus": CARD,
    "input_background_fill_hover": CARD,
    "input_border_color": BORDER,
    "input_border_color_focus": ACCENT,
    "input_border_color_hover": ACCENT,
    "input_placeholder_color": MUTED,
    # checkbox / radio
    "checkbox_background_color": CARD,
    "checkbox_background_color_selected": ACCENT,
    "checkbox_background_color_hover": CARD,
    "checkbox_border_color": BORDER,
    "checkbox_border_color_focus": ACCENT,
    "checkbox_border_color_hover": ACCENT,
    "checkbox_border_color_selected": ACCENT,
    "checkbox_label_background_fill": CARD,
    "checkbox_label_background_fill_selected": CARD,
    "checkbox_label_background_fill_hover": CARD,
    "checkbox_label_text_color": TEXT,
    "checkbox_label_border_color": BORDER,
    # primary button (single accent, solid - no gradient). Uses the deeper accent
    # so its WHITE label clears AA; hover lightens to the brand accent.
    "button_primary_background_fill": ACCENT_BTN,
    "button_primary_background_fill_hover": ACCENT,
    "button_primary_border_color": ACCENT_BTN,
    "button_primary_text_color": WHITE,
    # secondary button
    "button_secondary_background_fill": CARD,
    "button_secondary_background_fill_hover": BORDER,
    "button_secondary_border_color": BORDER,
    "button_secondary_text_color": TEXT,
    # cancel / stop button
    "button_cancel_background_fill": "#3a1f24",
    "button_cancel_background_fill_hover": "#4a262c",
    "button_cancel_border_color": "#5a2b32",
    "button_cancel_text_color": "#ffd9dc",
    # tables / code / logs
    "table_even_background_fill": PANEL,
    "table_odd_background_fill": CARD,
    "table_border_color": BORDER,
    "table_text_color": TEXT,
    "code_background_fill": CODE_BG,
    # status / feedback
    "error_background_fill": "#3a1f24",
    "error_border_color": "#5a2b32",
    "error_text_color": "#ffd9dc",
    "shadow_drop": "0 1px 2px rgba(0,0,0,0.4)",
    "shadow_drop_lg": "0 6px 20px rgba(0,0,0,0.45)",
}


def build_theme() -> gr.themes.Base:
    """Return the Apex dark theme with the palette pinned for BOTH color modes.

    A token gets its `_dark` twin set only when the theme actually defines one
    (a few - e.g. color_accent, shadow_drop - have no per-mode variant); those
    single-valued tokens apply to both modes already.
    """
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=("system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"),
    )
    both: dict[str, str] = {}
    for key, val in _TOKENS.items():
        if hasattr(theme, key):
            both[key] = val
        if hasattr(theme, f"{key}_dark"):
            both[f"{key}_dark"] = val
    theme.set(**both)
    return theme
