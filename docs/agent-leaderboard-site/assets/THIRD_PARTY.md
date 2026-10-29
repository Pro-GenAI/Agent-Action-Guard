# Bundled leaderboard assets

Served locally to avoid automatic third-party asset requests. Snapshot fetched
29 October 2026. Keep license files alongside any redistributed assets.

- Inter and Space Grotesk: unmodified Latin variable WOFF2 subsets from Google
  Fonts; SIL Open Font License 1.1, copyright/license in `fonts/*-OFL.txt`.
  Exact source URLs are recorded in the main website's
  `assets/font-sources.json` in this repository.
- Tailwind CSS **3.4.19**: generated static utilities in `tailwind.css`;
  MIT license in `tailwind-LICENSE`. Build inputs and command are in this site's
  directory and README. Source: <https://github.com/tailwindlabs/tailwindcss/tree/v3.4.19>.
  The old browser runtime and unused forms/container-query plugins are no longer
  loaded; no browser-side build or network request is needed.
- Provider marks: existing Simple Icons SVGs, bundled unmodified in `providers/`.
  Google, Anthropic, Alibaba Cloud, X, and Mistral were fetched from
  `https://cdn.simpleicons.org/<slug>/e3e2e7`; OpenAI, Microsoft, and IBM from
  `https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/<slug>.svg`.
  `providers/simple-icons-LICENSE` contains the upstream CC0 license and
  trademark-use notes. These are a checked-in snapshot; pages do not resolve
  `latest` or contact either CDN. Marks belong to their respective owners and
  do not imply endorsement. Source: <https://github.com/simple-icons/simple-icons>.
- Small UI icons are inline project SVG paths, replacing the externally loaded
  Material Symbols font.

Keep the shared `.provider-logo` styles and graceful image-error hiding.

`social-card.svg` is editable project artwork using the established shield
motif; `social-card.jpg` is its generated export for sharing previews. These
project assets retain the project's existing license and introduce no remote
font/image requests.
