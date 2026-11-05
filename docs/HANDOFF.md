# Handoff: wayne.is-a.dev

Not published: `exclude_docs` in mkdocs.yml keeps this file out of the site.

## State (2026-11-05)

- Deploys on push to main: `.github/workflows/deploy.yml` builds `cv/cv.tex` with XeLaTeX (xu-cheng/latex-action@v4), copies it to `docs/cv.pdf`, then `mkdocs gh-deploy`. The committed `docs/cv.pdf` is overwritten at deploy time; edit `cv/cv.tex`, not the PDF.
- Current role: Field Application Engineer, Taiwan AILabs (About, Experiences, llms.txt, JSON-LD in overrides/main.html, cv.tex).
- Deploys also run hourly (cron :17) so the Patches page follows GitHub. `hooks/patches.py` reads `https://prs.wayne.is-a.dev/api/contributions.json` at build time and fills the `<!-- patches:summary -->` and `<!-- patches:projects -->` markers in docs/patches.md. State and every count come from the feed; `patches.yml` only groups repos into projects, lists the featured PRs with short titles, and names PRs that landed outside the merge button (PyTorch ghstack). PRs into my own repos are skipped. A repo no project claims shows up as its own group. A featured PR that gets closed drops off with a build warning. If the feed is down the build fails and the last deploy stays live. Tests: `python -m unittest discover -s tests` (also run in CI).
- To feature a new PR: add its URL and short title under `picks`. Nothing else needs editing.
- Upstream count was 24 merged/landed on 2026-11-05 (the Patches page now computes it). The CV (`cv/cv.tex`) and About ("more than 20") still carry it by hand; recount for those with `gh api -X GET search/issues -f q='is:pr author:waynehacking8 is:merged -user:waynehacking8 -repo:is-a-dev/register -repo:chihmingyang/cs2680'`.
- MathJax (blog math pages) loads from jsDelivr with an SRI sha384; bumping the version means recomputing the hash from the npm tarball.

## Traps

- No custom response headers on GitHub Pages: HSTS, CSP and frame headers are missing and cannot be added here. Scanners will flag them.
- Copy rules: no emoji, no dash punctuation, no AI-template phrasing. Never mention RTX PRO 6000 or 96GB; say "Blackwell workstation".
- A locally built `site/` must not be committed; gh-deploy builds its own.
- GitHub disables scheduled workflows after 60 days without repo activity; a push re-enables them. The PR wall's PR_COUNT must stay above the real open plus merged total, or the Patches page undercounts.

## Open

- Nothing blocking.
