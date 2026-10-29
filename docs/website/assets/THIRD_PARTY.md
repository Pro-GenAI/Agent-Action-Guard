# Bundled website assets

Served locally to avoid automatic third-party asset requests. Snapshot fetched
29 October 2026. Keep license files alongside any redistributed assets.

- Manrope, Inter, Space Grotesk, and JetBrains Mono: unmodified Latin variable
  WOFF2 subsets from Google Fonts, under the SIL Open Font License 1.1. Each
  font's copyright and license is in `fonts/*-OFL.txt`. Exact stylesheet, font,
  and license source URLs are recorded in `font-sources.json`. Other characters
  use the existing system-font fallbacks.
- Highlight.js **11.11.1**: the existing cdnjs browser build, now copied to
  `vendor/highlight-11.11.1.min.js`. BSD 3-Clause license in
  `vendor/highlight-LICENSE`. Source:
  <https://github.com/highlightjs/highlight.js/tree/11.11.1>.
- `iceberg.jpg`, `demo.gif`, and `Workflow.gif`: copied from the project's existing
  root `assets/`. Project-authored material follows the repository license;
  third-party elements retain their own terms. The workflow icon source credits
  below are preserved, but a credit is not evidence of every required permission.
  Verify upstream rights before republishing or reusing those elements.

Do not replace local references with remote font, script, or image embeds.

`demo.webp` is a lossless optimized alternative to the existing `demo.gif`, with
the same 42 decoded frames and timing; the GIF remains the fallback. The frame
and file checksums are in `demo-optimization.json`. The original image credits
below apply to the optimized derivative too. `project-logo.jpg` copies the
existing root project logo for same-origin structured metadata. `social-card.svg`
is editable project artwork using the established shield motif; `social-card.jpg`
is its generated export. These project assets retain the project's license.

## Existing illustration credits

The project's existing workflow illustration uses icons credited in the root
`assets/README.md`; retain these credits with the copied GIF:

Image credits for icons:
- User: https://www.flaticon.com/free-icon/user_9131478   
- Robot: https://www.flaticon.com/free-icon/robot_18355220
- Data processing: https://www.flaticon.com/free-icon/data-processing_7017511
- Action: https://www.flaticon.com/free-icon/automation_2103800
- Response: https://www.flaticon.com/free-icon/fast-response_10748876
- Toolkit: https://www.flaticon.com/free-icon/tool-box_2276313
- Harmful action: https://www.flaticon.com/free-icon/danger_12211330
- Harmful query: https://www.flaticon.com/free-icon/abuse_4665375
