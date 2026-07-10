# UI-5 — GUI 2.0 design overhaul — build + gate record (2026-07-10)

**Status: DONE. Visual gate PASSED 24/24 (`PASS=True`), 151 tests pass.** Do NOT commit
(per WP instructions). Fixes the four user-reported defects: (1) not full-screen / wrong
aspect, (2) mixed black/white + black-text-on-black, (3) em-dashes, (4) verbose copy.

Machine-readable evidence (non-gitignored):
- `_infusion/ui5_visual_check.json` — full trace: 24 screens × {scanned nodes, contrast fails,
  em-dash flag, container width ratio}.
- `_infusion/ui5_visual_check.py` — re-runnable gate (`python _infusion/ui5_visual_check.py`).
- `docs/gui2_screens/*.png` — 24 screenshots (committable; PNGs are not gitignored outside `qa/`).

---

## 1. Root cause (found)

The wizard mixed TWO independent theme systems that could disagree:

- **Gradio `Soft` theme** styled its own components (inputs, dropdowns, buttons, accordions,
  markdown) via `.dark`-scoped CSS tokens that follow the browser's `prefers-color-scheme`.
- **`gui2/theme.css`** styled our custom HTML cards via `--apex-*` tokens that flipped under a
  SEPARATE `@media (prefers-color-scheme: light)` rule.

When the browser's color mode and Gradio's mode detection diverged, one surface rendered light
and its neighbour stayed dark. Concretely: a dark `--apex-panel` card (default value, no media
match) holding Gradio-light text (dark text) = **black text on black**; and Gradio-light white
panels sitting on our dark `.gradio-container` background = **the black/white mix**. Confirmed by
reading `launch()` (passed `theme=gr.themes.Soft()` + `css=_CSS`, both mode-reactive) and
`theme.css` (dark default + a `@media (prefers-color-scheme: light)` override block).

## 2. Token system chosen (one dark palette, pinned for BOTH modes)

`gui2/theme.py` (NEW) builds one theme. Gradio emits base tokens under `:root {}` and dark
tokens under `:root.dark {}` (`gradio/themes/base.py`). We set **each token AND its `_dark`
twin to the same dark value**, so whether or not the `.dark` class is present, every token
resolves to the same dark value — the browser's color preference **cannot** produce a mixed
render (no reload/`__theme` hack needed; verified light-emulated Step 1 renders fully dark).

Palette (single source; mirrored in `theme.css` `--apex-*`, defined once, no media flip):

| role | value | notes |
|---|---|---|
| app background | `#0e0f13` | body |
| panel / block | `#16181f` | cards, inputs' parents |
| elevated / input | `#1c1f28` | inputs, secondary buttons |
| border | `#2a2f3a` | |
| text primary | `#e7e9ee` | ~12:1 on app bg |
| text secondary | `#a3aab8` | ~7.5:1 on panel |
| accent | `#7c6cff` | borders / active step / stars / links (high contrast on dark) |
| accent (white-text) | `#6656e6` | primary buttons + active step badge; white clears AA (5.2:1) — white on `#7c6cff` is only 3.6:1 |
| good / warn | `#3ecf8e` / `#f0a84a` | stepper-done / flag border |

`theme.css` is now LAYOUT-ONLY (cards, stepper, quality grid, progress, full-width container).
Every inline-styled HTML surface already routed color through `--apex-*`; the only hardcoded
value fixed was the viewer caption (`opacity:0.65` → solid `--apex-muted`; iframe bg `#111` →
`#0b0c10`).

## 3. Full-width layout

`gr.Blocks(fill_width=True)` + `.gradio-container{max-width:100%!important;width:100%!important}`
fills the viewport; `padding: calc(max(20px,(100% - 1400px)/2))` (border-box) keeps content near
1400px centered on wide screens with **no** horizontal overflow. Measured container width ratio =
**1.000** on all 24 screens (>=0.95 required).

## 4. Copy sweep (what was simplified)

- **Em-dashes:** removed from EVERY user-visible string in `gui2/` (app.py, deliver.py, qc.py,
  runner.py). Total em-dash count in `gui2/` fell 151 → 109; the **109 remaining are all code
  comments/docstrings + the vendored `viewer/index.js` bundle** (separate iframe document), none
  rendered. The DOM scan confirms **0 U+2014 in `document.body.innerText`** on all 24 screens.
  Notable render-time fixes: `deliver.star_string` None-fallback `"—"→"-"`, and the quality-card /
  deliverable-card `"—"` placeholders `→ "-"`.
- **Verbose → terse (one sentence max):** Step-3 review paragraph (6 lines → 2), Advanced-drawer
  intro, live-preview blurb, detail-checkbox label ("Detail mode (sharper microstructure,
  slower — ~15 MP, pose refinement off)" → "Detail mode (sharper, slower)"), Utilities tab
  descriptions, mode/detail/status notes.
- **Evidence citations dropped from the UI:** the Advanced drawer info was
  `HELP + " · Evidence: " + EVIDENCE`; now just `HELP`. `defaults.EVIDENCE` stays in code.
- **Never-fabricate discipline kept:** the quality card still renders only fields that exist in
  the delivery record (honest flags, "not scored" where a source is absent).

## 5. Gate — playwright visual check (PASS 24/24)

`_infusion/ui5_visual_check.py` serves the REAL wizard (same mount as `gui2.app.launch`:
`build_theme()` + `theme.css` + `/viewer` + `/scenes`) and drives it in **both** emulated
browser color schemes (`prefers-color-scheme` light AND dark) at **1920×1080 and 2560×1440**.

- **Phase A (live server):** Step 1, Utilities (expanded), Advanced drawer (expanded) — real
  Gradio components.
- **Phase B (standalone faithful render):** Steps 2/3/4 custom surfaces (progress bar, stage
  list, quality card, deliverable cards, honest-flag box) rendered with the **identical** tokens
  — `build_theme()._get_theme_css()` (byte-for-byte what Gradio injects) + `theme.css` + the real
  `gui2.deliver`/`gui2.app` renderer output for `_dataset/UI4_SMOKE`. Necessary because Gradio
  **lazily renders `visible=False` columns** (verified: Steps 2/3/4 are absent from the initial
  DOM), so they can't be shown without a full pipeline run; their Gradio component chrome is the
  same components validated live in Phase A.

Automated DOM audit per screenshot: WCAG-AA contrast (>=4.5:1 normal, >=3:1 large text) on every
visible text node (computed color vs walked-up effective background, alpha-blended); zero U+2014
in body innerText; `.gradio-container` width >=95% of viewport.

| metric | before (old theme.css + Soft) | after |
|---|---|---|
| contrast failures (24 screens) | not measured pre-fix; the mixed-mode render produced black-on-black by construction | **0** |
| screens with em-dash in body text | (star/placeholder "—" rendered; verbose copy) | **0** |
| screens failing full-width (>=95%) | old `.gradio-container{max-width:1080px}` → ~0.56 at 1920 (**FAIL**) | **0** (ratio 1.000) |

`24/24 OK; PASS=True` (`ui5_visual_check.json`). Both light- and dark-emulated browsers render
identically dark (the whole point).

## 6. Files changed

| File | Change |
|---|---|
| `gui2/theme.py` (NEW) | `build_theme()` — one dark palette pinned for both modes + AA-safe button accent. |
| `gui2/theme.css` (rewritten) | Layout-only; `--apex-*` defined once (no media flip); full-width container + centered 1400px content. |
| `gui2/app.py` | `fill_width=True`; theme+css applied at `mount_gradio_app` (Gradio 6 moved them off the Blocks ctor); copy sweep; dropped Evidence from Advanced info; added `elem_id`s to step columns + HTML holders (additive, for the gate). |
| `gui2/deliver.py`, `gui2/qc.py`, `gui2/runner.py` | Em-dash removal in rendered strings / status / log lines. |
| `_infusion/ui5_visual_check.py` (NEW) | The playwright gate. |
| `docs/GUI_REDESIGN.md`, `THIRD_PARTY.md` | §5.5 build record + row; playwright dev-dep + theme provenance. |

## 7. Flags / levers

No new trainer flag, no runtime dependency, torch untouched. Production behavior of every
non-`gui2` module is unchanged. Playwright is a **dev/test-only** dependency (imported solely by
the gate script, never by the app or `tests/`). Theme is default and total (there is one theme);
no A/B flag applies to a visual-identity change, but the legacy `gui/` + `run.py` are untouched.

## 8. Server

Old server on :7861 killed; a fresh server with the new theme is left running on
**127.0.0.1:7861** (`python run2.py`).
