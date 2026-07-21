const { test, expect } = require('@playwright/test');

const base = process.env.PB_BASE_URL || 'http://127.0.0.1:8765';
const article = '/blog/rag-groundedness-guardrail/';

async function geometry(page) {
  return page.evaluate(() => {
    const rect = selector => {
      const element = document.querySelector(selector);
      const box = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        top: box.top,
        bottom: box.bottom,
        width: box.width,
        height: box.height,
        cssTop: style.top,
        padding: style.padding,
        margin: style.margin,
        transitionDuration: style.transitionDuration,
      };
    };
    return {
      y: scrollY,
      profile: rect('.pb-side'),
      bar: rect('.pb-nav-row'),
      title: rect('.pb-post-title'),
      profileVariable: parseFloat(getComputedStyle(document.documentElement)
        .getPropertyValue('--pb-mobile-profile-height')),
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    };
  });
}

for (const width of [320, 390, 768, 1023]) {
  test(`mobile article header follows pbb scroll geometry at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(`${base}${article}`, { waitUntil: 'networkidle' });

    await expect(page.locator('.pb-page-title')).toHaveText('Blog');
    await expect(page.locator('.pb-post-title')).toHaveText('The 0% RAG result failed the harder test');
    await expect(page.locator('.pb-nav-row')).toContainText('About');
    await expect(page.locator('.pb-nav-row')).not.toContainText('RAG groundedness');

    const top = await geometry(page);
    expect(top.overflow).toBe(false);
    expect(Math.abs(top.profile.height - top.profileVariable)).toBeLessThan(1);
    if (width <= 579) expect(Math.abs(top.profile.height - 129)).toBeLessThan(1);
    expect(top.bar.padding).toBe('20px 24px');
    expect(top.bar.margin).toBe('0px -24px 40px');
    expect(top.bar.top).toBeCloseTo(top.profile.bottom + 20, 0);
    expect(top.title.top).toBeCloseTo(top.bar.bottom + 40, 0);
    expect(top.bar.height).toBeGreaterThan(72);
    expect(top.bar.height).toBeLessThan(77);

    await page.evaluate(() => scrollTo(0, 80));
    await expect(page.locator('body')).toHaveClass(/pb-strip-hidden/);
    await page.waitForTimeout(300);
    let state = await geometry(page);
    expect(state.profile.bottom).toBeCloseTo(0, 0);
    expect(state.bar.cssTop).toBe('0px');

    await page.evaluate(() => scrollTo(0, 220));
    await page.waitForTimeout(300);
    state = await geometry(page);
    expect(state.bar.top).toBeCloseTo(0, 0);

    await page.evaluate(() => scrollTo(0, 420));
    await page.waitForTimeout(50);
    await page.evaluate(() => scrollTo(0, 320));
    await expect(page.locator('body')).not.toHaveClass(/pb-strip-hidden/);
    await page.waitForTimeout(300);
    state = await geometry(page);
    expect(state.profile.top).toBeCloseTo(0, 0);
    expect(state.bar.top).toBeCloseTo(state.profile.bottom, 0);
    expect(state.bar.top).toBeGreaterThanOrEqual(state.profile.bottom - 1);
    expect(state.overflow).toBe(false);

    await page.evaluate(() => scrollTo(0, 0));
    await expect(page.locator('body')).not.toHaveClass(/pb-strip-hidden/);
  });
}

test('desktop article label bar matches pbb sticky offset and structure', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}${article}`, { waitUntil: 'networkidle' });
  const top = await geometry(page);
  expect(top.bar.padding).toBe('20px 40px 20px 24px');
  expect(top.bar.margin).toBe('0px -40px 40px -24px');
  expect(top.bar.height).toBeGreaterThan(80);
  expect(top.bar.height).toBeLessThan(84);
  expect(top.title.top).toBeCloseTo(top.bar.bottom + 40, 0);

  await page.evaluate(() => scrollTo(0, 240));
  await page.waitForTimeout(300);
  const sticky = await geometry(page);
  expect(sticky.bar.top).toBeCloseTo(48, 0);
  expect(sticky.bar.cssTop).toBe('48px');
  await expect(page.locator('body')).not.toHaveClass(/pb-strip-hidden/);
});

test('reduced motion keeps pbb state changes without animation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}${article}`, { waitUntil: 'networkidle' });
  let state = await geometry(page);
  expect(state.profile.transitionDuration).toBe('0s');
  expect(state.bar.transitionDuration).toBe('0s');

  await page.evaluate(() => scrollTo(0, 220));
  await expect(page.locator('body')).toHaveClass(/pb-strip-hidden/);
  state = await geometry(page);
  expect(state.profile.bottom).toBeCloseTo(0, 0);
  expect(state.bar.top).toBeCloseTo(0, 0);
});

test('soft navigation and history restore the article label state', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}/blog/`, { waitUntil: 'networkidle' });
  await page.locator('.pb-post-card[href$="rag-groundedness-guardrail/"]').click();
  await expect(page).toHaveURL(new RegExp(`${article}$`));
  await expect(page.locator('.pb-page-title')).toHaveText('Blog');
  await expect(page.locator('.pb-post-title')).toBeVisible();

  await page.evaluate(() => scrollTo(0, 220));
  await expect(page.locator('body')).toHaveClass(/pb-strip-hidden/);
  await page.locator('.pb-page-title-link').click();
  await expect(page).toHaveURL(`${base}/blog/`);
  await expect(page.locator('body')).not.toHaveClass(/pb-strip-hidden/);

  await page.goBack({ waitUntil: 'networkidle' });
  await expect(page).toHaveURL(new RegExp(`${article}$`));
  await expect(page.locator('.pb-page-title')).toHaveText('Blog');
  const state = await geometry(page);
  expect(state.overflow).toBe(false);
});
