# Privacy and launch-feature review

Reviewed 29 October 2026. Scope: the shipped Python and TypeScript runtimes,
`docs/website/`, and `docs/agent-leaderboard-site/`. This records the implementation
and future release checks; it is not a certification of legal compliance.

The [six-point reel](https://www.instagram.com/reel/Dd8beLqSP-Y/) prompted this
review. Its penalty examples are not adopted as project policy: applicability
depends on the service, facts, jurisdiction, and current law.

| Area | Current implementation | Review before adding a related feature |
| --- | --- | --- |
| Children and signup | Static developer documentation and benchmark results; no account registration or age collection. | Assess audience and information collection before adding accounts or collecting children's data. Where applicable, implement parental notice, verifiable consent, access/deletion, and retention controls. A date-of-birth field alone is not a compliance solution. |
| Third-party fonts and assets | Fonts, syntax highlighting, styles, illustrations, and provider marks are served from each site's own origin. Font licenses and third-party attribution accompany the files. | Keep automatic browser asset requests on the same origin. Review licenses and data transfers before adding embeds or remote assets. |
| Analytics and session replay | No analytics, advertising pixels, session replay, keystroke recording, or persistent browser storage in the authored sites. Pointer effects only change local visual styles. | Review purpose, minimization, masking, retention, and applicable consent before adding tracking. Do not record action payloads, credentials, or form text. |
| Marketing email | No mailing-list signup or marketing sender. `send_email` examples describe screened agent tools; they are not a project mailing system. | Review commercial-message rules; provide accurate sender/subject information, a valid postal address, and a working opt-out with timely suppression. Keep marketing separate from service messages. |
| Subscription renewals | No checkout, billing, paid subscription, or recurring charge. External provider/API fees are governed by those providers. | Disclose price, billing interval, trial conversion, and renewal/cancellation terms before purchase; obtain applicable affirmative consent, preserve evidence, and support required reminders and straightforward cancellation. |
| Copyright and uploads | No public user-content hosting/upload service. Local JSON/JSONL CLI input and classification HTTP requests are processed as actions, not published as user content. | Review rights, reporting/takedown processes, repeat-infringer policy, and applicable safe-harbor requirements before hosting user uploads. A footer does not register a designated agent; designation and renewals require an operator to use the Copyright Office's system. |

## Runtime data flow

Both packages perform classification locally. Without API configuration, default
embedding assets are downloaded and cached locally; the asset host receives the
download request, not the action being classified. Set `AAG_EMBED_ONNX` or
`AAG_EMBED_GGUF` to pre-provisioned local files/directories for local embedding
without model downloads. Optional GGUF bindings and other dependencies must also
be installed/provisioned in advance for an offline environment.

An explicit GGUF source takes precedence over ONNX, and an explicit ONNX source
takes precedence over API settings. Otherwise, `EMBED_MODEL_NAME`,
`EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, or `OPENAI_API_KEY` can select an
OpenAI-compatible embedding API. In that mode action text is sent to the
configured endpoint. Review that provider's processing, retention, location,
and agreement before sending sensitive actions. Normal classification does not
send telemetry to the project maintainers.

The optional evaluation/LLM comparison workflows intentionally send action
payloads to configured model providers and write result caches. CLI output,
application logs, proxies, framework integrations, and operator-hosted HTTP
services can also retain data. Operators control their deployment's access,
logging, retention, deletion, and privacy notices; the website notice does not
cover every downstream application. Keep the HTTP API's loopback default unless
an access-controlled deployment is intentionally configured.

## Maintenance and verification

- Keep `/privacy/` linked from both sites and documentation, and keep public
  routes in the respective sitemaps.
- Keep `/terms/` linked alongside privacy information; its payment/refund section
  describes the project's current free, no-checkout scope. Keep cookie information
  and data-request/deletion routes discoverable from every public page.
- Run `python3 -m unittest discover -s docs/website/tests -v` after website edits.
  The tests check same-origin asset references, bundled licenses, privacy links,
  metadata, and the existing benchmark/documentation snapshot.
- Run `node --test docs/website/tests/tabs.test.cjs` for the shipped homepage
  script's keyboard selection, focus, panel visibility, and install-copy tests.
- Preview both sites, including mobile, syntax highlighting, provider icons, and
  reduced-motion behavior. Rebuild the leaderboard CSS after utility changes
  using the pinned command in `docs/agent-leaderboard-site/README.md`.
- Revisit this inventory and the notices when adding accounts, tracking,
  messaging, payments, uploads, or a hosted API. Operational registrations and
  jurisdiction-specific decisions require the operator; do not invent a postal
  address, a registered DMCA agent, or a blanket compliance claim in code.

## Follow-up: twenty-area launch checklist

The [second reel](https://www.instagram.com/reel/Ddb8FK7S0wy/) prompted a broader
review on 29 October 2026. Its suggested controls are applied according to the
features actually shipped, rather than assuming every site collects data or
takes payments. This is an implementation inventory, not legal certification.

| # | Review area | Current outcome and evidence |
| --- | --- | --- |
| 1 | Privacy information | Both sites publish `/privacy/`, linked from their homepage and policy navigation. It distinguishes site hosting, local classification, API embedding, and evaluation caches. |
| 2 | Conditions of use | Both sites publish `/terms/`, linked to the existing CC BY 4.0 license and third-party notices. No new click-through contract or account terms are introduced. |
| 3 | Payment/refund scope | `/terms/#refunds` explains free project access, no checkout/subscriptions, and independent provider charges. Future paid features require their own terms. |
| 4 | Browser-storage disclosure | `/privacy/#cookies` describes the authored sites' lack of cookies, persistent storage, and tracking SDKs, plus the hosting-provider distinction. |
| 5 | Tracking consent | No optional tracking is enabled, so no fictitious consent toggle/banner is presented. Review any new tracking before loading it, including applicable opt-in and withdrawal controls. |
| 6 | Intake permissions | No signup, newsletter, or data-entry form exists. Data requests open the visitor's mail client; nothing is submitted automatically. Any future intake must disclose its purpose and separate optional marketing consent. |
| 7 | Data minimization | No age, account, or consent-profile database is added. Data-request instructions ask for only enough information to locate the record and discourage secrets/identity documents in the initial request. |
| 8 | Vendor/code inventory | All website assets are same-origin. Fonts, Highlight.js, generated Tailwind CSS, and provider marks have local provenance/license notes. Runtime third-party data flow remains documented above. |
| 9 | Fair interactions | Ordinary links/buttons, visible keyboard focus, and accessible tab selection; no preselected purchase, forced registration, countdown, or consent wall. The inert leaderboard report button is replaced with a working methodology link. |
| 10 | Cost transparency | Free project access is distinguished from separately chosen paid model APIs/hosting in the terms. There are no hidden project fees or auto-renewals. |
| 11 | Recognition provenance | Existing mentions/stars link to their source and are described as mentions or stars, not customer reviews. Homepage and terms explicitly avoid implying endorsement by people or employers. |
| 12 | Evidence for claims | Removed zero-latency, universal throughput, unsupported quantization/edge-accuracy, output-interception, and swarm-coordination claims. Headline metrics use the existing 97.87% HarmActions accuracy and 19.66 ms/action CPU benchmark with limitations and warm-up exclusion. Leaderboard statements are bounded to tested prompts/models. |
| 13 | Image descriptions | Meaningful images retain descriptive alt text. Repeated provider marks use empty alt text and are marked decorative beside the visible model/provider name; decorative SVG icons are hidden from assistive technology. |
| 14 | Readable contrast | Increased dim homepage text and leaderboard secondary labels; representative text/focus palette pairs have automated 4.5:1 checks. This does not claim a complete WCAG conformance audit. |
| 15 | Keyboard access | Every public page has a skip link and focusable main target. Tab groups have associated focusable panels, one active tab stop, arrow wrapping, Home/End support, and native Tab/Enter/Space behavior. Focused reveal content remains visible; leaderboard content is readable without JavaScript. |
| 16 | Project identity/contact | Privacy and terms identify the open-source project, public repository, maintainer first name, and existing public email. No company registration, postal address, or registered DMCA-agent status is invented. |
| 17 | Children's information | These are general developer/research resources with no registration or age collection. Review audience, parental notice/consent, and children's-data obligations before adding relevant features; a checkbox is not a substitute. |
| 18 | Messaging opt-out | No marketing email sender or list exists. Any future commercial sender must satisfy applicable identification, opt-out, suppression, and address requirements described in the six-area review. |
| 19 | Asset rights/attribution | Local font/library licenses, source URLs, provider-mark notes, and the existing workflow-image icon credits are retained in each site's `assets/THIRD_PARTY.md` and linked from policy pages. Existing asset provenance is preserved; this inventory does not establish rights absent from upstream material. |
| 20 | Personal-data requests | `/privacy/#data-requests` offers an email route for access/correction/deletion, distinguishes maintainer correspondence from local files and provider-held records, and explains that the static sites have no account/action-history store. Operators handle their own deployments' retention and deletion. |

The accessibility changes cover the authored pages and controls inspected here.
Continue manual testing with assistive technologies and real workloads; do not
market this checklist or the classifier as a guarantee of accessibility, safety,
or legal compliance. Revisit identity, jurisdiction, operational response
processes, and upstream asset rights with the operator as the service changes.

## Follow-up: eleven-point publishing review

The [fourth reel](https://www.instagram.com/reel/Ddv_d_sPx4l/) prompted this
follow-up on 29 October 2026. Most of its controls were already implemented by
the earlier reviews. This pass checks the actual static-site behavior and fixes
remaining disclosure, evidence, and interaction gaps. It does not certify
compliance with every local law.

| # | Reel topic | Current implementation and follow-up |
| --- | --- | --- |
| 1 | Privacy policy | Both origins publish the same notice. `/privacy/#data-collected` now names the data, recipient, purpose, and retention responsibility for hosting, voluntary email, and runtime use. |
| 2 | Terms and conditions | Both `/terms/` pages retain the existing project license, scope, limitations, and public contact. No account agreement or new click-through contract is added. |
| 3 | Cookie policy | `/privacy/#cookies` states the authored pages' tracking/storage behavior and distinguishes GitHub hosting. The operator chose to keep the sites tracking-free. |
| 4 | Refund policy | `/terms/#refunds` explains free access, no project checkout/subscription, and separately purchased provider services. No fictitious paid offering is introduced. |
| 5 | Cookie and form consent | No optional trackers or site forms exist. `/privacy/#third-parties` explains that email links open the visitor's mail app and require their own send action; no mailing-list enrollment occurs. Any future form/embed needs a feature-specific review. |
| 6 | What user data is collected | The new definition list separates provider-held hosting records, correspondence the user sends, and action data controlled by the package operator or configured API provider. It makes no unsupported fixed retention/deletion promise. |
| 7 | Third-party embeds | Local assets and ordinary outbound links are preserved. No social feed, video iframe, analytics SDK, map, remote font, or third-party form is added. Regression tests reject automatic remote asset loads and authored network/storage calls. |
| 8 | Alt text and colour contrast | Meaningful images have alternatives; repeated decorative marks are hidden beside visible names. Existing contrast checks remain, and new copy-status text uses the tested muted-text palette. This is not a full WCAG audit. |
| 9 | Keyboard use and clear controls | Existing skip links and roving tabs remain. Copy controls now announce success/failure through visible polite status regions. A false/throwing legacy copy command never reports success; temporary fields are cleaned up, focus restored, and duplicate pending copies suppressed. Permission errors explain manual copying. Tests cover both buttons, fallback modes, retries, concurrency, and missing optional elements. |
| 10 | Reviews, claims, and transparency | Removed three star cards and one like card because profile/post links did not establish the named interaction. Removed employer titles from the retained source-linked mention cards. A bibliography reference is labeled as such. Benchmark values and their limitations are unchanged; the remaining social sources still need operator verification if access-gated. |
| 11 | Image rights and applicable local requirements | Terms and asset notes now distinguish project licensing from third-party image/trademark permissions. Source credits are retained but are not described as proof of permission. Operator review is still needed for upstream rights and jurisdiction-specific duties; no invented registration, address, consent record, or blanket legal guarantee is added. |

Run the standard-library website suite and Node interaction tests after changing
these controls, plus the isolated browser suite when available. Keep this file's
public copies and the two policy bodies/styles synchronized. Test discovery is
not browser execution; native Safari, assistive-technology checks, gated social
sources, and jurisdiction-specific operational decisions remain separate checks.

## Follow-up: project identity and app-store disclosures

The [launch reel](https://www.instagram.com/reel/DeUKCUDtprn/) asks for terms,
privacy information, a trademark search, and accurate app-store labels. Both
sites already provide `/terms/` and `/privacy/`. Their data inventory separates
GitHub Pages request metadata, voluntary email correspondence, local package
processing, configured embedding APIs, and optional evaluation providers. The
project does not currently offer a native app or publish an app-store privacy
label. A downstream application must assess its own integrated code, providers,
and data collection before making a store disclosure; the website's no-tracking
statement is not a universal label for applications using the packages.

**Trademark clearance remains open.** A name or domain search alone does not
establish rights. Before adopting or expanding the name in a jurisdiction, the
operator should identify target jurisdictions and goods/services, search exact
and similar word and design marks plus relevant unregistered uses, record the
results and dates, and seek qualified advice where needed. The project does not
claim a trademark registration or a completed clearance search. See the [USPTO
clearance-search guidance](https://www.uspto.gov/trademarks/search/comprehensive-clearance-search-similar-trademarks)
and [Apple's app privacy details](https://developer.apple.com/app-store/app-privacy-details/).

## Primary references

- [FTC COPPA six-step compliance plan](https://www.ftc.gov/business-guidance/resources/childrens-online-privacy-protection-rule-six-step-compliance-plan-your-business)
- [Google Fonts: privacy and font requests](https://fonts.googleblog.com/2022/11/your-privacy-and-google-fonts.html)
- [FTC CAN-SPAM guide](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business)
- [California Attorney General: automatic renewal](https://www.oag.ca.gov/news/press-releases/attorney-general-bonta-issues-consumer-alert-california%E2%80%99s-automatic-renewal-law)
- [U.S. Copyright Office designated-agent FAQ](https://www.copyright.gov/dmca-directory/faq.html)
- [GitHub privacy statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement)
- [W3C text contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
- [W3C accessible tabs pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/)
- [ICO cookies and similar technologies](https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/cookies-and-similar-technologies/)
- [ICO erasure requests](https://ico.org.uk/for-the-public/your-right-to-get-your-data-deleted/)
- [FTC evidence for advertising claims](https://www.ftc.gov/business-guidance/resources/advertising-faqs-guide-small-business)

- [Creative Commons BY 4.0: attribution and additional rights](https://creativecommons.org/licenses/by/4.0/)
