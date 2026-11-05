# Handoff: wayne.is-a.dev

Not published: `exclude_docs` in mkdocs.yml keeps this file out of the site.

## State (2026-11-05)

- Deploys on push to main: `.github/workflows/deploy.yml` builds `cv/cv.tex` with XeLaTeX (xu-cheng/latex-action@v4), copies it to `docs/cv.pdf`, then `mkdocs gh-deploy`. The committed `docs/cv.pdf` is overwritten at deploy time; edit `cv/cv.tex`, not the PDF.
- Current role: Field Application Engineer, Taiwan AILabs (About, Experiences, llms.txt, JSON-LD in overrides/main.html, cv.tex).
- Upstream count: 24 merged/landed (GitHub search, merged, own repos and coursework excluded, plus the PyTorch NCCL fix landed as c3c33fd). Patches page, CV and About ("more than 20") agree. Recount when it moves: `gh api -X GET search/issues -f q='is:pr author:waynehacking8 is:merged -user:waynehacking8 -repo:is-a-dev/register -repo:chihmingyang/cs2680'`.
- MathJax (blog math pages) loads from jsDelivr with an SRI sha384; bumping the version means recomputing the hash from the npm tarball.

## Traps

- No custom response headers on GitHub Pages: HSTS, CSP and frame headers are missing and cannot be added here. Scanners will flag them.
- Copy rules: no emoji, no dash punctuation, no AI-template phrasing. Never mention RTX PRO 6000 or 96GB; say "Blackwell workstation".
- A locally built `site/` must not be committed; gh-deploy builds its own.

## Open

- Nothing blocking.
