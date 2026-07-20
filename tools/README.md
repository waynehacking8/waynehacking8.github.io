# Font subset

`lxgw-name.woff2` is LXGW WenKai TC cut down to the three glyphs the site
sets in the CJK name (邱 偉 誠). It is **inlined as a data URI** in
`docs/stylesheets/extra.css`, so this copy is the source of record, not a
served asset.

Regenerate after changing the name:

```sh
# full family (13MB) from the Google Fonts CSS `src:` URL
curl -sL -H 'User-Agent: Mozilla/5.0' \
  'https://fonts.googleapis.com/css2?family=LXGW+WenKai+TC&display=swap'
curl -sL -o lxgw-full.ttf '<the gstatic .ttf URL from that CSS>'

python -m fontTools.subset lxgw-full.ttf --text='邱偉誠' --flavor=woff2 \
  --layout-features='' --no-hinting --desubroutinize \
  --output-file=lxgw-name.woff2

base64 -w0 lxgw-name.woff2   # paste into the @font-face src in extra.css
```

Why inlined rather than linked: the Google Fonts `@import` silently fell back
to system sans on a real iPhone over cellular. A separate request — to any
host — is a request that can be skipped. At 1.8KB the font rides along with
the CSS and there is nothing left to fail.
