# Handoff: wayne.is-a.dev

Not published: `exclude_docs` in mkdocs.yml keeps this file out of the site.

## State (2026-11-05)

- Deploys on push to main and hourly: `.github/workflows/deploy.yml` runs the unit tests, writes cv/stats.tex, builds `cv/cv.tex` with XeLaTeX, copies it to `docs/cv.pdf`, runs `mkdocs build --strict`, checks internal links with lychee, and publishes the `site/` folder as a Pages artifact (Pages source: GitHub Actions; gh-pages is no longer used). Only the deploy job holds `pages: write` and `id-token: write`. Actions are pinned to commit SHAs and requirements.txt to exact versions; Dependabot (`.github/dependabot.yml`) proposes weekly updates. Rollback to the old flow: `gh api -X PUT repos/waynehacking8/waynehacking8.github.io/pages -f build_type=legacy -f source[branch]=gh-pages -f source[path]=/` and restore the previous deploy.yml.
- Current role: Field Application Engineer, Taiwan AILabs (About, Experiences, llms.txt, JSON-LD in overrides/main.html, cv.tex).
- Deploys also run hourly (cron :17) so the Patches page follows GitHub. `hooks/patches.py` reads `https://prs.wayne.is-a.dev/api/contributions.json` at build time and fills the `<!-- patches:summary -->` and `<!-- patches:projects -->` markers in docs/patches.md. State and every count come from the feed; `patches.yml` only groups repos into projects, lists the featured PRs with short titles, and names PRs that landed outside the merge button (PyTorch ghstack). PRs into my own repos are skipped. A repo no project claims shows up as its own group. A featured PR that gets closed drops off with a build warning. If the feed is down the build fails and the last deploy stays live. Tests: `python -m unittest discover -s tests` (also run in CI).
- To feature a new PR: add its URL and short title under `picks`. Nothing else needs editing.
- Every upstream count on the site is live, from the same fetch: Patches page, About (`<!-- upstream:merged -->` in docs/index.md), llms.txt (rewritten in `on_post_build`, since it is a static file), the JSON-LD description (`config.extra.upstream.merged` in overrides/main.html), and the CV. For the CV, CI runs `python hooks/patches.py cv/stats.tex` before LaTeX; cv.tex does `\input{stats}` and uses `\upstreammerged`, `\upstreamwhere`, `\upstreamgroups`. The committed cv/stats.tex is only a snapshot for local LaTeX builds. Do not type a count into any page again. Cross-check: `gh api -X GET search/issues -f q='is:pr author:waynehacking8 is:merged -user:waynehacking8 -repo:is-a-dev/register -repo:chihmingyang/cs2680'` plus the landed PyTorch PR.
- MathJax (blog math pages) loads from jsDelivr with an SRI sha384; bumping the version means recomputing the hash from the npm tarball.

## Traps

- No custom response headers on GitHub Pages: HSTS, CSP and frame headers are missing and cannot be added here. Scanners will flag them.
- Copy rules: no emoji, no dash punctuation, no AI-template phrasing. Never mention RTX PRO 6000 or 96GB; say "Blackwell workstation".
- A locally built `site/` must not be committed; CI builds its own.
- The TeXLive image is pinned by digest in deploy.yml (`docker_image`); Dependabot does not see it, so bump it by hand when the CV needs a newer TeX. lychee remaps `https://wayne.is-a.dev/...` onto the build, so self-links are checked; directory links need an index.html.
- requirements.txt was frozen on Windows, so it lists colorama; harmless on Linux. The gh-pages branch is no longer deployed from and can be deleted.
- GitHub disables scheduled workflows after 60 days without repo activity; a push re-enables them. The PR wall's PR_COUNT must stay above the real open plus merged total, or the Patches page undercounts.

## Open

- Nothing blocking.
