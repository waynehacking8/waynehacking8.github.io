# TOC and Patches verification log

## Scope

- Fix article table-of-contents state across hard loads, soft navigation, and browser history.
- Improve desktop and mobile TOC readability without changing article heading accents or global typography.
- Replace the Patches dashboard-like PR-wall block with one restrained editorial link and an inline RSS link.
- Preserve the existing sidebar identity, theme behavior, responsive layout, images and captions, filters, and unrelated navigation.

## Decisions

- Keep the profile sidebar DOM persistent.
- Render the desktop TOC with each replaceable `.pb-main`, then move that single current node into a dedicated persistent sidebar slot during initialization. Clear the slot and disconnect its observer before every main-content swap.
- Use one viewport-appropriate TOC as the scroll-spy source so exactly one current link receives `aria-current="location"`.
- Keep native hash navigation and add heading scroll margins for the sticky navigation/profile strip.
- Use one compact PR-wall link with a subtle live marker, visible URL, inline counts, and a separate text-level RSS link below it.

## Verification matrix

- [x] Strict MkDocs build
- [x] Hard-load article desktop TOC
- [x] Article to Patches and other non-article navigation clears TOC
- [x] Patches to article restores the correct TOC
- [x] Article-to-article navigation replaces headings without stale links
- [x] Back/forward restores current TOC
- [x] Mobile/desktop expose only their intended TOC
- [x] Hash navigation, sticky offset, scroll spy, and reduced motion
- [x] Repeated navigation disconnects observers and does not multiply global handlers
- [x] Required Patches widths have no horizontal overflow
- [x] PR-wall/RSS destinations and visible keyboard focus
- [x] Light/dark axe checks
- [x] No console errors or failed first-party requests
- [x] Existing images, captions, filters, sidebar, heading accents, theme toggle, and navigation
- [x] Required screenshots inspected
- [x] Lighthouse article and Patches reports reviewed
- [x] Final diff inspected
- [ ] Production deployment and cache-busted assets verified

## Evidence

- `python -m mkdocs build --strict`: passed.
- Playwright: 101 passed in the final combined local run.
- Axe: no violations in either light or dark mode for the changed components; broader existing route audits also passed.
- Required screenshots: `test-results/toc-patches-final/`; all eight inspected for hierarchy, wrapping, density, alignment, overflow, and theme treatment.
- Lighthouse 13.4.0 desktop:
  - Article: Performance 93, Accessibility 100, Best Practices 100, SEO 100, CLS 0.
  - Patches: Performance 97, Accessibility 100, Best Practices 100, SEO 100, CLS 0.
- Cache versions: `extra.css?v=20260720.23` and `pb-filter.js?v=20260720.7`.
- Lighthouse produced valid JSON reports; its Windows launcher logged a non-fatal temporary-profile cleanup warning after report generation.
