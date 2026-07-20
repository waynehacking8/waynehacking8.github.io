# CJK name font subset

The site sets exactly three glyphs in a display face — 邱 偉 誠 — so the family
is cut down to those and **inlined as a data URI** in
`docs/stylesheets/extra.css` under the private family name `PB Name CJK`
(private so it cannot collide with a locally installed copy of the same family).
`name-cjk.woff2` here is the source of record, not a served asset.

Current face: **Noto Serif TC, weight 900** — 1572 bytes subset. Chosen over the
previous kai script (LXGW WenKai TC) for weight and presence; it also survives
the 20px the name is actually set at, which the handwriting candidates did not.
The `@font-face` declares `font-weight: 400` because that is the weight the
elements ask for — the outlines themselves are the 900 cut.

## Regenerate

Needed whenever the name changes: new characters have **no glyphs at all**
otherwise, because of the `unicode-range` on the `@font-face`.

```sh
# 1. get the full family. The Google Fonts CSS API hands back a .ttf URL when
#    called with a plain User-Agent; grab the URL out of the CSS.
curl -sL -H 'User-Agent: Mozilla/5.0' \
  'https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@900&display=swap'
curl -sL -o full.ttf '<the fonts.gstatic.com .ttf URL from that CSS>'

# 2. cut it to just the glyphs in the name
python -m fontTools.subset full.ttf --text='邱偉誠' --flavor=woff2 \
  --layout-features='' --no-hinting --desubroutinize \
  --output-file=name-cjk.woff2

# 3. inline it, and update the unicode-range to the new codepoints
python -c "import base64;print(base64.b64encode(open('name-cjk.woff2','rb').read()).decode())"
```

Then bump `extra_css`'s `?v=` in `mkdocs.yml`.

## Two things worth knowing before swapping the face

**Simplified-only brush fonts are a trap.** Ma Shan Zheng, Zhi Mang Xing and
Long Cang all look great and all lack 偉 and 誠 — only 邱 renders and the rest
fall back, which is the mixed-face bug this whole thing started with. Check
coverage with `fontTools` before getting attached to a candidate.

**Inline rather than link.** The earlier Google Fonts `@import` silently fell
back to system sans on a real iPhone over cellular, while every desktop engine
reported the font loaded — a second request, to any host, is a request that can
be dropped. At ~1.5KB the font rides along inside the stylesheet and there is
nothing left to fail.

Verify a swap with the font hosts blocked at the network layer, and compare
**glyph metrics**, not `document.fonts.check()` — that tests the Latin string
"BESbwy" and answers yes even when the CJK subset never arrived.
