# Name font subsets

Both halves of the name in the sidebar are set in subset, inlined faces:
`name-cjk.woff2` (邱偉誠) and `name-latin-400.woff2` ("Wei-Cheng Chiu").
Everything below applies to both.

## CJK — 邱偉誠

The site sets exactly three glyphs in a display face — 邱 偉 誠 — so the family
is cut down to those and **inlined as a data URI** in
`docs/stylesheets/extra.css` under the private family name `PB Name CJK`
(private so it cannot collide with a locally installed copy of the same family).
`name-cjk.woff2` here is the source of record, not a served asset.

Current face: **莫大毛筆體 / Bakudai Bold** — a brush hand, 2340 bytes subset.
SIL OFL 1.1, after 青柳衡山's 衡山毛筆フォント
(<https://github.com/max32002/bakudaifont>). The upstream family carries a
Reserved Font Name, which is why the family is renamed here; licence text in
`OFL.txt` alongside.

The brief landed here after two passes: Noto Serif TC 900 was 霸氣 but not
artistic, 衡山行書 (`max32002/masafont`, also OFL) was 太草. This one keeps the
brush strokes upright and unjoined, so it still resolves at name size.

Brush glyphs sit small inside their em box, so the name is set a couple of px
larger than the Latin would need — 23px on desktop, `clamp(12px, 4.3vw, 17px)`
on phones. Raising it further starts to crowd the theme toggle; the name has a
44px right gutter reserved for exactly that.

## Latin — Wei-Cheng Chiu

**Alex Brush Regular** (SIL OFL 1.1), subset to the ten characters the name
uses, 1.88KB, under the private family `PB Name Latin`. This handwritten script
was explicitly selected to pair with the brush CJK. **This is a deliberate
departure from pbb.**

Alex Brush has only one weight, so the original thin/bold split is intentionally
absent and synthetic bold is disabled. Its compact x-height keeps the name at
33px on desktop and `clamp(16px, 5.7vw, 23px)` on phones. Regenerate with the
same `fontTools.subset` command, using `--text='Wei-ChngCu '`.

## Regenerate

Needed whenever the name changes: new characters have **no glyphs at all**
otherwise, because of the `unicode-range` on the `@font-face`.

```sh
# 1. get the full family (19MB)
curl -sL -o full.ttf \
  https://raw.githubusercontent.com/max32002/masafont/master/tw/MasaFont-Bold.ttf

# 2. cut it to just the glyphs in the name
python -m fontTools.subset full.ttf --text='邱偉誠' --flavor=woff2 \
  --layout-features='' --no-hinting --desubroutinize \
  --output-file=name-cjk.woff2

# 3. inline it, and update the unicode-range to the new codepoints
python -c "import base64;print(base64.b64encode(open('name-cjk.woff2','rb').read()).decode())"
```

Then bump `extra_css`'s `?v=` in `mkdocs.yml`.

## Two things worth knowing before swapping the face

**Simplified fonts are a trap, in two different ways.** Ma Shan Zheng, Zhi Mang
Xing and Long Cang simply lack 偉 and 誠 — only 邱 renders and the rest fall back,
which is the mixed-face bug this whole thing started with. Worse, the 王漢宗
faces (`cghio/wangfonts`) *pass* a cmap check at those codepoints and then draw
the **Simplified** 伟/诚 anyway, because they are GB-to-Big5 conversions. A
coverage check is not enough — always render the three glyphs and look.

**Inline rather than link.** The earlier Google Fonts `@import` silently fell
back to system sans on a real iPhone over cellular, while every desktop engine
reported the font loaded — a second request, to any host, is a request that can
be dropped. At ~2.4KB the font rides along inside the stylesheet and there is
nothing left to fail.

Verify a swap with the font hosts blocked at the network layer, and compare
**glyph metrics**, not `document.fonts.check()` — that tests the Latin string
"BESbwy" and answers yes even when the CJK subset never arrived.
