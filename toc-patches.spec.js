const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const base = process.env.PB_BASE_URL || 'http://127.0.0.1:8765';
const articleA = '/blog/inkling-975b-architecture/';
const articleB = '/blog/kimi-k3-architecture/';
const articleASlug = 'inkling-975b-architecture/';
const articleBSlug = 'kimi-k3-architecture/';

async function goto(page, path) {
  return page.goto(`${base}${path}`, { waitUntil: 'networkidle' });
}

async function clickNav(page, name, path) {
  await page.locator('.pb-nav-link', { hasText: name }).click();
  await expect(page).toHaveURL(new RegExp(`${path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`));
  await expect(page.locator('body')).not.toHaveClass(/pb-is-navigating/);
}

test('hard-loaded article has one current desktop TOC', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const response = await goto(page, articleA);
  expect(response.status()).toBe(200);
  await expect(page.locator('.pb-sidebar-toc-slot .pb-article-toc--desktop')).toBeVisible();
  await expect(page.locator('.pb-main > .pb-article-toc--desktop')).toHaveCount(0);
  await expect(page.locator('.pb-article-toc--mobile')).toBeHidden();
  await expect(page.locator('.pb-article-toc [aria-current="location"]')).toHaveCount(1);
});

test('article to Patches clears every article TOC immediately', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  await clickNav(page, 'Patches', '/patches/');
  await expect(page.locator('.pb-article-toc')).toHaveCount(0);
  await expect(page.locator('.pb-sidebar-toc-slot')).toBeEmpty();
  await expect(page.locator('[aria-current="location"]')).toHaveCount(0);
});

test('every non-article soft-navigation target remains free of article TOCs', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  const destinations = [
    ['Patches', '/patches/'],
    ['About', '/'],
    ['Experiences', '/experiences/'],
    ['Publications', '/publications/'],
    ['Projects', '/projects/'],
    ['Talks', '/talks/'],
    ['Blog', '/blog/'],
  ];
  for (const [name, path] of destinations) {
    await clickNav(page, name, path);
    await expect(page.locator('.pb-article-toc')).toHaveCount(0);
  }
});

test('Patches to article restores the article-specific TOC', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, '/patches/');
  await clickNav(page, 'Blog', '/blog/');
  await page.locator(`.pb-post-card[href$="${articleASlug}"]`).click();
  await expect(page).toHaveURL(new RegExp(`${articleA}$`));
  await expect(page.locator('.pb-sidebar-toc-slot .pb-article-toc--desktop')).toBeVisible();
  const hrefs = await page.locator('.pb-article-toc--desktop a').evaluateAll(links => links.map(link => link.hash));
  expect(hrefs.length).toBeGreaterThan(1);
  expect(await page.evaluate(ids => ids.every(id => document.getElementById(decodeURIComponent(id.slice(1)))), hrefs)).toBe(true);
});

test('direct article-to-article soft navigation replaces all stale TOC links', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  const oldHrefs = await page.locator('.pb-article-toc--desktop a').evaluateAll(links => links.map(link => link.hash));
  await page.locator('.pb-page-title-link').evaluate((link, href) => link.setAttribute('href', href), articleB);
  await page.locator('.pb-page-title-link').click();
  await expect(page).toHaveURL(new RegExp(`${articleB}$`));
  const newHrefs = await page.locator('.pb-article-toc--desktop a').evaluateAll(links => links.map(link => link.hash));
  expect(newHrefs).not.toEqual(oldHrefs);
  expect(await page.evaluate(ids => ids.every(id => document.getElementById(decodeURIComponent(id.slice(1)))), newHrefs)).toBe(true);
  for (const stale of oldHrefs.filter(href => !newHrefs.includes(href))) {
    await expect(page.locator(`.pb-article-toc--desktop a[href="${stale}"]`)).toHaveCount(0);
  }
});

test('browser back and forward restore the correct TOC state', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  await clickNav(page, 'Patches', '/patches/');
  await page.goBack();
  await expect(page).toHaveURL(new RegExp(`${articleA}$`));
  await expect(page.locator('.pb-article-toc--desktop')).toBeVisible();
  await page.goForward();
  await expect(page).toHaveURL(/\/patches\/$/);
  await expect(page.locator('.pb-article-toc')).toHaveCount(0);
});

test('mobile exposes only the expandable TOC with readable touch targets', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await goto(page, articleA);
  await expect(page.locator('.pb-article-toc--desktop')).toBeHidden();
  const mobile = page.locator('.pb-article-toc--mobile');
  await expect(mobile).toBeVisible();
  await expect(mobile).not.toHaveAttribute('open', '');
  const summaryMetrics = await mobile.locator('summary').evaluate(element => {
    const style = getComputedStyle(element);
    return { fontSize: parseFloat(style.fontSize), height: element.getBoundingClientRect().height };
  });
  expect(summaryMetrics.fontSize).toBeGreaterThanOrEqual(13);
  expect(summaryMetrics.height).toBeGreaterThanOrEqual(44);
  await mobile.locator('summary').click();
  await expect(mobile).toHaveAttribute('open', '');
  expect(await mobile.locator('a').first().evaluate(link => link.getBoundingClientRect().height)).toBeGreaterThanOrEqual(40);
});

test('desktop exposes only one visible, readable TOC', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  await expect(page.locator('.pb-article-toc:visible')).toHaveCount(1);
  const sizes = await page.locator('.pb-article-toc--desktop').evaluate(toc => ({
    label: parseFloat(getComputedStyle(toc.querySelector('p')).fontSize),
    primary: parseFloat(getComputedStyle(toc.querySelector(':scope > ol > li > a')).fontSize),
    nested: parseFloat(getComputedStyle(toc.querySelector('ol ol a')).fontSize),
  }));
  expect(sizes.label).toBeGreaterThanOrEqual(12);
  expect(sizes.primary).toBeGreaterThanOrEqual(13);
  expect(sizes.nested).toBeGreaterThanOrEqual(12);
});

test('TOC links retain a shareable hash and clear the sticky navigation', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  const link = page.locator('.pb-article-toc--desktop a').nth(1);
  const hash = await link.getAttribute('href');
  await link.click();
  await expect(page).toHaveURL(new RegExp(`${hash.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`));
  const position = await page.evaluate(id => {
    const heading = document.getElementById(decodeURIComponent(id.slice(1)));
    const nav = document.querySelector('.pb-nav-row');
    return { top: heading.getBoundingClientRect().top, navBottom: nav.getBoundingClientRect().bottom };
  }, hash);
  expect(position.top).toBeGreaterThanOrEqual(position.navBottom - 1);
  await expect(page.locator(`[id="${decodeURIComponent(hash.slice(1))}"]`)).toBeInViewport();
});

test('scroll spy assigns aria-current to exactly the scrolled section', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  const links = page.locator('.pb-article-toc--desktop a');
  const index = Math.min(3, (await links.count()) - 1);
  const hash = await links.nth(index).getAttribute('href');
  await page.evaluate(id => document.getElementById(id).scrollIntoView(), decodeURIComponent(hash.slice(1)));
  await expect.poll(async () => page.locator('.pb-article-toc [aria-current="location"]').count()).toBe(1);
  await expect(page.locator('.pb-article-toc [aria-current="location"]')).toHaveAttribute('href', hash);
});

test('repeated soft navigation leaves one observer and does not multiply global handlers', async ({ page }) => {
  await page.addInitScript(() => {
    const NativeObserver = window.IntersectionObserver;
    const tracker = { created: 0, disconnected: 0, active: 0, listeners: {} };
    window.__pbTracker = tracker;
    window.IntersectionObserver = function (callback, options) {
      const observer = new NativeObserver(callback, options);
      const isArticleToc = options && options.rootMargin === '-120px 0px -65% 0px';
      const disconnect = observer.disconnect.bind(observer);
      let active = true;
      if (isArticleToc) {
        tracker.created += 1;
        tracker.active += 1;
      }
      observer.disconnect = () => {
        if (active && isArticleToc) {
          active = false;
          tracker.disconnected += 1;
          tracker.active -= 1;
        }
        return disconnect();
      };
      return observer;
    };
    window.IntersectionObserver.prototype = NativeObserver.prototype;
    const add = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, listener, options) {
      if ((this === window || this === document) && ['click', 'popstate', 'scroll'].includes(type)) {
        tracker.listeners[type] = (tracker.listeners[type] || 0) + 1;
      }
      return add.call(this, type, listener, options);
    };
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await goto(page, articleA);
  const baseline = await page.evaluate(() => ({ ...window.__pbTracker.listeners }));
  for (let index = 0; index < 3; index += 1) {
    await clickNav(page, 'Patches', '/patches/');
    await expect.poll(() => page.evaluate(() => window.__pbTracker.active)).toBe(0);
    await clickNav(page, 'Blog', '/blog/');
    await page.locator(`.pb-post-card[href$="${articleASlug}"]`).click();
    await expect(page).toHaveURL(new RegExp(`${articleA}$`));
    await expect.poll(() => page.evaluate(() => window.__pbTracker.active)).toBe(1);
  }
  const final = await page.evaluate(() => window.__pbTracker);
  expect(final.active).toBe(1);
  expect(final.created - final.disconnected).toBe(1);
  expect(final.listeners).toEqual(baseline);
});

for (const width of [320, 360, 375, 390, 430, 768, 1024, 1440]) {
  test(`Patches has no horizontal overflow at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await goto(page, '/patches/');
    expect(await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth)).toBe(false);
  });
}

test('PR-wall and RSS destinations are correct and keyboard focus is visible', async ({ page }) => {
  await goto(page, '/patches/');
  const wall = page.locator('.pb-pr-wall-main');
  const rss = page.locator('.pb-pr-wall-rss');
  await expect(wall).toHaveAttribute('href', 'https://prs.wayne.is-a.dev');
  await expect(rss).toHaveAttribute('href', 'https://prs.wayne.is-a.dev/feed.xml');
  for (const link of [wall, rss]) {
    await link.focus();
    await expect(link).toBeFocused();
    const outline = await link.evaluate(element => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: parseFloat(style.outlineWidth) };
    });
    expect(outline.style).not.toBe('none');
    expect(outline.width).toBeGreaterThanOrEqual(2);
  }
});

for (const scheme of ['default', 'slate']) {
  test(`changed components pass axe in ${scheme} mode`, async ({ page }) => {
    await goto(page, '/patches/');
    await page.evaluate(value => document.body.setAttribute('data-md-color-scheme', value), scheme);
    await page.waitForTimeout(250);
    expect((await new AxeBuilder({ page }).include('.pb-pr-wall').analyze()).violations).toEqual([]);
    await goto(page, articleA);
    await page.evaluate(value => document.body.setAttribute('data-md-color-scheme', value), scheme);
    await page.waitForTimeout(250);
    expect((await new AxeBuilder({ page }).include('.pb-article-toc').analyze()).violations).toEqual([]);
  });
}

test('soft navigation has no console errors or failed first-party requests', async ({ page }) => {
  const errors = [];
  const failed = [];
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('requestfailed', request => {
    if (new URL(request.url()).origin === new URL(base).origin) failed.push(request.url());
  });
  await goto(page, articleA);
  await clickNav(page, 'Patches', '/patches/');
  await clickNav(page, 'Blog', '/blog/');
  await page.locator(`.pb-post-card[href$="${articleBSlug}"]`).click();
  await expect(page).toHaveURL(new RegExp(`${articleB}$`));
  expect(errors).toEqual([]);
  expect(failed).toEqual([]);
});

test('sidebar, filters, images, captions, headings, theme, and navigation still work', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await goto(page, '/blog/');
  await expect(page.locator('.pb-side')).toBeVisible();
  await expect(page.locator('.pb-post-image img')).toHaveCount(10);
  await page.getByRole('button', { name: 'Security', exact: true }).click();
  await expect(page.locator('.pb-post-card:visible')).toHaveCount(3);
  await page.getByRole('button', { name: 'Security', exact: true }).click();
  await page.getByRole('button', { name: /switch to .* mode/i }).click();
  await page.locator(`.pb-post-card[href$="${articleASlug}"]`).click();
  await expect(page.locator('.pb-article-hero img')).toBeVisible();
  await expect(page.locator('.pb-article-hero figcaption a')).toHaveAttribute('href', /^https:\/\//);
  await expect(page.locator('.pb-main h2').first()).toHaveCSS('border-bottom-width', '2px');
  await expect(page.locator('.pb-article-toc--mobile')).toBeVisible();
});
