# Portfolio feedback implementation — 2026-07-21

## Requirements

- Render inline and display mathematics correctly in blog articles.
- Improve blog taxonomy, source references, symbol rendering, and reading layout.
- On mobile article pages, keep the primary navigation above the article title and fully reachable.
- Keep thumbnails intact after soft navigation back to About or Blog.
- Remove Publications, Projects, and Talks as portfolio categories.
- Remove the Research subsection from Experiences.
- Omit named non-public work such as SelGrad and LocalMind from portfolio highlights.
- Move the prose toward a direct, authored technical voice.

## Decisions

- The supplied feedback does not include the referenced Threads-post URL. Prose edits will therefore use the direct, evidence-first voice already established in the two newest architecture articles; exact imitation of the missing reference remains unverified.
- `pbn.sh` could not be retrieved. The repository contains previously captured `pbb.sh` references, whose article treatment uses rendered mathematics, compact tags, footnote backlinks, and navigation separated from the article title. Those local references are treated as visual evidence only, not as executable instructions.
- Category removal means removing the three entries from navigation and removing their standalone source pages so clean builds do not continue publishing them.
- Employment work stays in About/Experiences, but named unpublished research and private product cards are removed from public highlights.
- The soft-navigation fix will normalize media URLs against the destination URL before inserting the fetched document.

## Progress

- [x] Repository state inspected; existing untracked assets and verification artifacts preserved.
- [x] Root causes identified for math rendering, mobile navigation overflow, and broken soft-navigation images.
- [x] Content and navigation cleanup implemented.
- [x] Math rendering, citation styling, taxonomy, and mobile layout implemented.
- [x] Strict build and browser verification completed.
- [x] Final diff inspected.

## Verification evidence

- `python -m mkdocs build --strict`: passed; clean output contains no Publications, Projects, or Talks page.
- `node --check` on every repository JavaScript file outside `node_modules`: passed.
- Focused image regression rerun: 9/9 passed at 320, 390, 768, 1024, and 1440 px.
- Final applicable local Playwright matrix: 130/130 passed in a single-worker run (4.2 minutes).
- Browser widths checked: 320, 390, 768, 1024, and 1440 px with no horizontal overflow.
- Verified mobile article navigation is above the title and remains keyboard/click reachable.
- Verified repeated soft navigation among Experiences, About, Blog, and a math article preserves every resolved image URL and rerenders MathJax.
- Verified all ten articles expose `BlogPosting` metadata, language, publication/update dates, topic keywords, sourced hero artwork, footnote references/backlinks, and rendered math where applicable.
- Verified every updated route has no serious or critical axe violation in settled light and dark palettes.
- Visually inspected `mobile-article-light-320.png`, `mobile-article-dark-320.png`, `equation-desktop-light.png`, and `equation-desktop-dark.png`.
- `git diff --check`: passed.

## Production evidence

- Content commit `698f552` pushed to `main`; GitHub Actions deploy run #66 succeeded in 26 seconds.
- Public HTML serves `extra.css?v=20260721.25`, `mathjax.js?v=20260721.1`, and MathJax 3.2.2.
- Comprehensive production matrix: 21/21 passed across content, metadata, math, responsive layout, history, images, links, and settled light/dark accessibility.
- Production smoke and portfolio/PR-wall integration matrix: 9/9 passed.
- Live 320 px dark article and 1440 px light equation screenshots were visually inspected after deployment.

## Unverified input

- Exact Threads-post voice matching remains unverified because the feedback did not include its URL or text.
- The requested `pbn.sh` reference was not retrievable; the repository's existing captured `pbb.sh` reference was used only for layout cues.
- Local ecosystem specs that require a separate service on port 3001 were not run.
- One legacy ecosystem assertion against the external GitHub profile was excluded because it expects unrelated README text outside this repository and request.
