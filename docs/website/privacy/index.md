# Privacy & data use

**Canonical HTML page:** [Read the original version](https://action-guard.pro-genai.com/privacy/)

> This is the Markdown edition of the public page. The linked HTML page is the canonical version for search engines and the interactive website.

Updated 29 October 2026

This notice describes the static Agent Action Guard website, documentation, and HarmActionsEval leaderboard at `action-guard.pro-genai.com` and `agent-leaderboard.github.io`, and the data flow of the project's Python and JavaScript packages.

## Visiting the websites

The authored pages do not use analytics, advertising trackers, session recording, keystroke logging, cookies, or persistent browser storage. Fonts, styles, syntax highlighting, illustrations, and provider logos are served from the same website origin. Animations and pointer effects run in your browser and are not sent to the project.

GitHub Pages serves these sites and receives normal request information, including your IP address and browser/request metadata. Hosting processing and retention are governed by [GitHub's privacy statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement). This notice does not promise that the hosting provider keeps no logs.

Links to GitHub, package registries, model providers, papers, and other external sites take you to services with their own privacy practices. The authored pages do not embed those services or automatically load their assets.

## What information is handled, and by whom?

Website request information

GitHub Pages receives IP addresses and browser/request metadata to deliver these pages. GitHub controls its hosting records and retention under its linked privacy statement; the project does not run a visitor analytics database.

Email you choose to send

The maintainer and email services handling the message receive your sender address, message, and any attachments you send. The maintainer uses correspondence to answer the request and retains it as needed for that purpose and applicable obligations. There is no fixed deletion deadline promised here. Use the [data-request route](https://action-guard.pro-genai.com/privacy/#data-requests) for access, correction, or deletion questions.

Actions processed by the packages

Normal local classification does not send action payloads to the maintainer. Your chosen embedding API can receive action text, and optional LLM evaluation can send actions to a model provider. Your application controls saved actions, logs, and result caches. See [runtime data flow](https://action-guard.pro-genai.com/privacy/#runtime-data) before processing sensitive inputs.

## External links, embeds, and consent

There are no embedded videos, social feeds, maps, third-party forms, or remote font/CDN assets in the authored pages. An external link is a navigation choice; the destination's data practices apply after you open it. Provider logos and local demo animations do not connect to the named provider.

Email links open your mail application. The site does not submit a message or add you to a marketing list; sending remains your choice. There is no site form or marketing-consent checkbox. If forms or third-party embeds are introduced, review their data recipients and purpose before loading them, and provide any applicable consent and withdrawal controls.

## Cookies and similar technologies

The authored sites set no cookies or browser storage and load no analytics or advertising SDKs. There are no optional tracking technologies to enable, so the pages have no cookie-consent banner or stored consent preference. GitHub Pages still processes hosting requests as described above. If tracking is introduced, its purpose, providers, retention, and any required consent controls must be reviewed before it is enabled.

## Accounts, payments, and uploads

These static sites have no account signup, newsletter registration, checkout, subscriptions, or user-content upload feature. They are developer and research resources, not services directed to children under 13, and do not request ages or dates of birth.

## Using the Python or JavaScript package

Action classification runs locally. By default, embedding model assets are downloaded and cached on your device; the model host receives asset requests, not your action text. To use pre-provisioned local embedding assets, set `AAG_EMBED_GGUF` or `AAG_EMBED_ONNX` to a local file or directory. Remote model sources still require downloads.

If you configure an embedding API using `EMBED_MODEL_NAME`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, or `OPENAI_API_KEY`, action text is sent to the configured embedding provider unless an explicit local-source setting takes precedence. GGUF takes precedence over ONNX; ONNX takes precedence over API settings. Review the provider's privacy and retention terms before submitting sensitive data.

Normal classification does not send telemetry to the project maintainers. Optional LLM evaluation workflows send actions to your configured model provider and persist result caches. Your application, CLI output, integrations, logs, or operator-hosted HTTP service may also retain action data. You control access and retention in your deployment; this notice does not replace its privacy notice.

## Access, correction, and data deletion requests

You can [request access, correction, or deletion of personal information sent to the project maintainer](mailto:praneeth.vad@gmail.com?subject=Agent%20Action%20Guard%20data%20request). Describe the information or page concerned and provide only enough detail to locate it. Do not send passwords, API keys, or identity documents with your initial request. Requests will be reviewed according to the data involved and applicable obligations.

The static sites have no user accounts or action-history database to delete. For local package use, you control your action files, CLI output, application logs, and evaluation result caches; review and remove your own saved copies when no longer needed. Downloaded model assets under `~/.cache/agent-action-guard/` are separate from those action/results files. A request to the maintainer does not remove files on your device or data held by your hosting or embedding provider; contact the relevant operator for those records.

## Project details and copyright concerns

Agent Action Guard is the open-source project maintained by [Pro-GenAI on GitHub](https://github.com/Pro-GenAI/Agent-Action-Guard). Project contact: Praneeth, [praneeth.vad@gmail.com](mailto:praneeth.vad@gmail.com). These sites do not offer a paid hosted service. See the [project terms and payment/refund information](https://action-guard.pro-genai.com/terms/).

Contact the project maintainer at [praneeth.vad@gmail.com](mailto:praneeth.vad@gmail.com) for privacy questions or concerns about project material. Include the relevant page or repository path, and avoid sending secrets or unnecessary personal information. Correspondence is used to address your request and may be retained as needed for that purpose. This contact is not represented as a registered DMCA agent.

The project's [CC BY 4.0 license](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/LICENSE.md) applies to project material as specified there. Bundled fonts and libraries retain their accompanying [third-party licenses and credits](https://action-guard.pro-genai.com/assets/THIRD_PARTY.md); provider marks belong to their respective owners.

We will update this notice when the project's data practices change. See the repository's [privacy and launch-feature review](https://action-guard.pro-genai.com/assets/privacy-review.md) for the current implementation inventory.
