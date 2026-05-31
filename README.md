# waynehacking8.github.io

Personal website / portfolio for **Wei Cheng (Wayne) Chiu** — built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

Structure: About · Experiences · Publications · Projects · Talks · Patches · Blog.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve            # http://127.0.0.1:8000
```

## Deploy
A push to `main` triggers `.github/workflows/deploy.yml`, which runs `mkdocs gh-deploy`
to build the site to the `gh-pages` branch. GitHub Pages is configured to serve from
`gh-pages`. The published site is https://waynehacking8.github.io/.
