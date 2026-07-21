# pbb.sh mobile article scroll parity — 2026-07-21

## Requirement

- Match pbb.sh's mobile article profile and label-bar positioning.
- On downward scroll, retract the profile and move the sticky label bar to the viewport top.
- On upward scroll, restore the profile and move the label bar beneath it.
- Preserve the behavior through soft navigation, browser history, theme changes, and reduced-motion preferences.

## Measured reference at 390 × 844

- pbb.sh profile height: 129 px.
- pbb.sh label-bar height: 74.4 px.
- At `scrollY = 0`, the label bar follows the profile in document flow.
- Downward movement beyond 50 px adds `body.header-hidden`; the profile translates by its full height and the label bar transitions from `top: 129px` to `top: 0`.
- Upward movement greater than 5 px removes `header-hidden`; the profile returns and the label bar transitions back to `top: 129px`.
- Both transitions use approximately 0.25 seconds with an ease curve.

## Original mismatch

- The portfolio waits until `scrollY > 140` before retracting.
- Only `.pb-side` moves; `.pb-nav-row` always uses `top: 0`.
- When scrolling upward, the restored profile overlaps the label bar instead of pushing it beneath the profile.
- The current 390 px article label bar is about 88.3 px tall because its navigation and title are stacked.

## Verification checklist

- [x] Profile and label bar use one measured height variable.
- [x] Downward and upward transitions match pbb.sh thresholds and geometry.
- [x] Article label-bar structure and height match the reference.
- [x] No overlap or horizontal overflow at 320, 390, 768, and 1023 px.
- [x] Soft navigation and history preserve one scroll listener and correct geometry.
- [x] Reduced motion removes animation without disabling retraction.
- [x] Strict build, focused Playwright, and full applicable local suite pass.
- [x] Production deployment and production browser suite pass.

## Local verification evidence

- Strict MkDocs build: passed.
- Focused scroll parity suite: 7/7 passed.
- Full applicable Playwright suite: 137/137 passed after correcting the desktop heading anchor offset for the taller sticky row.
- Measured local 390 x 844 geometry: 129 px profile and 74.39 px label bar; label-bar `top` transitions from 129 px to 0 and back to 129 px.
- Corrected top, downward-scroll, and upward-scroll screenshots inspected in `test-results/pbb-scroll-parity-final/`.
- `node --check` and `git diff --check`: passed.

## Production verification evidence

- Deployment workflow `29793227669`: completed successfully.
- Production assets: `extra.css?v=20260721.26` and `pb-filter.js?v=20260721.9`.
- Production browser suite: 12/12 passed across 320, 390, 768, 1023, and 1440 px, including reduced motion, soft navigation, article ordering, and TOC hash positioning.
- Measured production 390 x 844 geometry matches local and reference values: 129 px profile and 74.39 px label bar, with restored/hidden label-bar tops of 129/0 px.
- Production top, downward-scroll, and upward-scroll screenshots inspected in `test-results/pbb-scroll-parity-production/`; no overlap or horizontal overflow was present.
