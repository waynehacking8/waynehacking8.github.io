const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const base = process.env.PB_BASE_URL || 'http://127.0.0.1:8765';
const articles = [
  { slug: 'inkling-975b-architecture', language: 'zh-Hant', tags: ['Architecture', 'MoE', 'Multimodal'], math: true },
  { slug: 'kimi-k3-architecture', language: 'zh-Hant', tags: ['Architecture', 'Linear Attention', 'MoE'], math: true },
  { slug: 'notes-trtllm-triton-serving', language: ['en', 'zh-Hant'], tags: ['Serving', 'TensorRT-LLM', 'Triton'] },
  { slug: 'nccl-nvlink-bandwidth', language: ['en', 'zh-Hant'], tags: ['NCCL', 'Distributed', 'NVLink', 'LLM Serving'], math: true },
  { slug: 'rag-groundedness-guardrail', language: ['en', 'zh-Hant'], tags: ['RAG', 'Evaluation', 'Grounding', 'LLM'], math: true },
  { slug: 'notes-federated-learning-dp', language: ['en', 'zh-Hant'], tags: ['Privacy', 'Federated Learning', 'Differential Privacy'] },
  { slug: 'notes-cuda-tensor-core-gemm', language: ['en', 'zh-Hant'], tags: ['CUDA', 'Tensor Cores', 'Performance'] },
  { slug: 'meta-rl', language: 'zh-Hant', tags: ['Machine Learning', 'Reinforcement Learning', 'Survey'], math: true },
  { slug: 'pp-intro', language: 'zh-Hant', tags: ['CUDA', 'Parallel Programming', 'Performance'] },
  { slug: 'pentest-intro', language: 'zh-Hant', tags: ['Security', 'Penetration Testing', 'Methodology'], math: true },
];
const routes = ['/', '/experiences/', '/patches/', '/blog/', ...articles.map(article => `/blog/${article.slug}/`)];

async function expectNoHorizontalOverflow(page) {
  const widths = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(widths.scroll).toBe(widths.client);
}

async function expectDecodedImages(page, selector) {
  const failures = await page.locator(selector).evaluateAll(images => Promise.all(images.map(image =>
    new Promise(resolve => {
      const source = image.currentSrc || image.src;
      const probe = new Image();
      probe.onload = () => resolve(probe.naturalWidth > 0 ? null : source);
      probe.onerror = () => resolve(source);
      probe.src = source;
    })))).then(results => results.filter(Boolean));
  expect(failures).toEqual([]);
}

test('removed sections are absent from source-visible routes and navigation', async ({ page, request }) => {
  await page.goto(`${base}/`, { waitUntil: 'networkidle' });
  const navigation = await page.locator('.pb-nav-tabs').innerText();
  expect(navigation).not.toMatch(/Publications|Projects|Talks/);
  expect(await page.locator('.pb-main').innerText()).not.toMatch(/SelGrad|LocalMind/);

  await page.goto(`${base}/experiences/`, { waitUntil: 'networkidle' });
  expect(await page.locator('.pb-main h2').allTextContents()).not.toContain('Research');
  expect(await page.locator('.pb-main').innerText()).not.toMatch(/SelGrad|LocalMind/);

  for (const path of ['/publications/', '/projects/', '/talks/']) {
    expect((await request.get(`${base}${path}`)).status()).toBe(404);
  }
});

test('blog topic filters exactly match card taxonomy and persist in history', async ({ page }) => {
  await page.goto(`${base}/blog/`, { waitUntil: 'networkidle' });
  const expectedTopics = ['Architecture', 'Serving', 'CUDA', 'Distributed', 'RAG', 'Privacy', 'ML', 'Security'];
  await expect(page.locator('.pb-filters button')).toHaveText(expectedTopics);
  await expect(page.locator('.pb-post-card')).toHaveCount(10);

  const cardTags = await page.locator('.pb-post-card').evaluateAll(cards =>
    cards.map(card => card.dataset.tags.split(',')));
  const known = new Set(expectedTopics);
  expect(cardTags.every(tags => tags.length && tags.every(tag => known.has(tag)))).toBe(true);
  expect(expectedTopics.every(topic => cardTags.some(tags => tags.includes(topic)))).toBe(true);

  for (const topic of expectedTopics) {
    const button = page.getByRole('button', { name: topic, exact: true });
    await button.click();
    await expect(button).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('.pb-post-card:visible')).toHaveCount(
      cardTags.filter(tags => tags.includes(topic)).length);
    expect(new URL(page.url()).searchParams.getAll('tag')).toEqual([topic]);
    await button.click();
  }

  await page.getByRole('button', { name: 'CUDA', exact: true }).click();
  await page.locator('.pb-post-card:visible').first().click();
  await expect(page).toHaveURL(/\/blog\/[^?]+\/$/);
  await page.goBack({ waitUntil: 'networkidle' });
  await expect(page).toHaveURL(/\/blog\/\?tag=CUDA$/);
  await expect(page.locator('.pb-post-card:visible')).toHaveCount(2);
});

for (const article of articles) {
  test(`${article.slug} has complete metadata, references, media, and math`, async ({ page }) => {
    const consoleErrors = [];
    const failedFirstParty = [];
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    page.on('requestfailed', request => {
      if (new URL(request.url()).origin === new URL(base).origin) failedFirstParty.push(request.url());
    });

    const response = await page.goto(`${base}/blog/${article.slug}/`, { waitUntil: 'networkidle' });
    expect(response.status()).toBe(200);
    await expectDecodedImages(page, '.pb-article-hero img');
    await expect(page.locator('.pb-article-hero figcaption a')).toHaveAttribute('href', /^https:\/\//);
    expect(await page.locator('a.footnote-ref').count()).toBeGreaterThan(0);
    await expect(page.locator('.footnote')).toHaveCount(1);
    expect(await page.locator('.footnote-backref').count()).toBeGreaterThan(0);

    const articleTags = await page.locator('meta[property="article:tag"]').evaluateAll(meta =>
      meta.map(item => item.content));
    expect(articleTags).toEqual(article.tags);
    const schema = JSON.parse(await page.locator('script[type="application/ld+json"]').last().textContent());
    expect(schema['@type']).toBe('BlogPosting');
    expect(schema.inLanguage).toEqual(article.language);
    expect(schema.keywords).toEqual(article.tags);
    expect(schema.datePublished).toMatch(/^20\d\d-\d\d-\d\d$/);
    expect(schema.image).toMatch(/^https:\/\//);

    if (article.math) {
      expect(await page.locator('.arithmatex').count()).toBeGreaterThan(0);
      await page.locator('.arithmatex mjx-container').first().waitFor({ timeout: 15000 });
      expect(await page.locator('.arithmatex mjx-container').count()).toBeGreaterThan(0);
    }
    await expectNoHorizontalOverflow(page);
    expect(consoleErrors).toEqual([]);
    expect(failedFirstParty).toEqual([]);
  });
}

for (const width of [320, 390, 768, 1440]) {
  test(`every article is readable with ordered navigation at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    for (const article of articles) {
      await page.goto(`${base}/blog/${article.slug}/`, { waitUntil: 'domcontentloaded' });
      await expectNoHorizontalOverflow(page);
      await expectDecodedImages(page, '.pb-article-hero img');
      if (width < 1024) {
        const order = await page.evaluate(() => {
          const nav = document.querySelector('.pb-nav-row--article').getBoundingClientRect();
          const title = document.querySelector('.pb-post-title').getBoundingClientRect();
          return { navBottom: nav.bottom, titleTop: title.top, navWidth: nav.width };
        });
        expect(order.navBottom).toBeLessThanOrEqual(order.titleTop + 1);
        expect(order.navWidth).toBeGreaterThanOrEqual(width - 13);
        const links = page.locator('.pb-nav-row--article .pb-nav-link');
        for (let index = 0; index < await links.count(); index += 1) {
          await links.nth(index).scrollIntoViewIfNeeded();
          await expect(links.nth(index)).toBeVisible();
        }
      }
    }
  });
}

test('all blog surfaces remain readable in dark mode', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  for (const path of ['/blog/', ...articles.map(article => `/blog/${article.slug}/`)]) {
    await page.goto(`${base}${path}`, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => document.body.setAttribute('data-md-color-scheme', 'slate'));
    await expectNoHorizontalOverflow(page);
    const colors = await page.evaluate(() => {
      const style = getComputedStyle(document.querySelector('.pb-main'));
      return { background: style.backgroundColor, color: style.color };
    });
    expect(colors.background).not.toBe(colors.color);
  }
});

test('soft navigation repeatedly preserves About and Blog images and rerenders math', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto(`${base}/experiences/`, { waitUntil: 'networkidle' });
  await page.locator('.pb-nav-link', { hasText: 'About' }).click();
  await expect(page).toHaveURL(`${base}/`);
  await expectDecodedImages(page, '.pb-main img');

  await page.locator('.pb-nav-link', { hasText: 'Blog' }).click();
  await expect(page).toHaveURL(`${base}/blog/`);
  await expectDecodedImages(page, '.pb-post-image img');
  await page.locator('.pb-post-card[href$="rag-groundedness-guardrail/"]').click();
  await expect(page).toHaveURL(/\/blog\/rag-groundedness-guardrail\/$/);
  await page.locator('.arithmatex mjx-container').first().waitFor({ timeout: 15000 });

  await page.goBack({ waitUntil: 'networkidle' });
  await expectDecodedImages(page, '.pb-post-image img');
  await page.goBack({ waitUntil: 'networkidle' });
  await expect(page).toHaveURL(new RegExp(`^${base.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/(?:#.*)?$`));
  await expectDecodedImages(page, '.pb-main img');
  await page.goForward({ waitUntil: 'networkidle' });
  await expectDecodedImages(page, '.pb-post-image img');
});

test('all first-party page links resolve', async ({ page, request }) => {
  const links = new Set();
  for (const route of routes) {
    await page.goto(`${base}${route}`, { waitUntil: 'domcontentloaded' });
    const found = await page.locator('a[href]').evaluateAll((anchors, origin) => anchors
      .map(anchor => new URL(anchor.getAttribute('href'), location.href))
      .filter(url => url.origin === origin && !url.pathname.startsWith('/assets/'))
      .map(url => `${url.origin}${url.pathname}${url.search}`), new URL(base).origin);
    found.forEach(link => links.add(link));
  }
  for (const link of links) {
    const response = await request.get(link);
    expect(response.status(), `${link} returned ${response.status()}`).toBeLessThan(400);
  }
});

test('every updated route has no serious or critical axe violations in both themes', async ({ page }) => {
  for (const route of routes) {
    await page.goto(`${base}${route}`, { waitUntil: 'domcontentloaded' });
    for (const scheme of ['default', 'slate']) {
      await page.evaluate(value => document.body.setAttribute('data-md-color-scheme', value), scheme);
      // Audit settled theme colors, not the intentional 200 ms color transition.
      await page.waitForTimeout(250);
      const result = await new AxeBuilder({ page }).analyze();
      const blocking = result.violations.filter(violation =>
        violation.impact === 'serious' || violation.impact === 'critical');
      expect(blocking, `${route} ${scheme}: ${JSON.stringify(blocking, null, 2)}`).toEqual([]);
    }
  }
});

test('visual evidence covers mobile navigation and equations in both themes', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 900 });
  await page.goto(`${base}/blog/rag-groundedness-guardrail/`, { waitUntil: 'networkidle' });
  await page.screenshot({ path: 'test-results/portfolio-feedback/mobile-article-light-320.png' });
  await page.evaluate(() => document.body.setAttribute('data-md-color-scheme', 'slate'));
  await page.screenshot({ path: 'test-results/portfolio-feedback/mobile-article-dark-320.png' });

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(`${base}/blog/nccl-nvlink-bandwidth/`, { waitUntil: 'networkidle' });
  await page.locator('.arithmatex').first().scrollIntoViewIfNeeded();
  await page.screenshot({ path: 'test-results/portfolio-feedback/equation-desktop-light.png' });
  await page.evaluate(() => document.body.setAttribute('data-md-color-scheme', 'slate'));
  await page.screenshot({ path: 'test-results/portfolio-feedback/equation-desktop-dark.png' });
});
