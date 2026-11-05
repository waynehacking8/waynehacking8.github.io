"""Build the Patches page from live PR state.

The page used to hard-code which PRs were merged and how many were in review,
so it drifted every time a PR moved. Now patches.yml only says how repos group
into projects and which PRs to feature; state, counts and the date come from
the PR wall's public feed (prs.wayne.is-a.dev) at build time. The deploy
workflow rebuilds on a schedule, so the page follows GitHub without edits.

A failed fetch fails the build on purpose: the last good deploy stays live,
which beats publishing a page with no data.
"""

from __future__ import annotations

import html
import json
import logging
import urllib.request
from pathlib import Path

import yaml

log = logging.getLogger("mkdocs.hooks.patches")

CONFIG = Path(__file__).resolve().parent.parent / "patches.yml"
SUMMARY_MARK = "<!-- patches:summary -->"
PROJECTS_MARK = "<!-- patches:projects -->"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "waynehacking8.github.io build"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _bucket(state: str) -> str | None:
    if state == "merged":
        return "merged"
    if state in ("open", "draft"):
        return "review"
    return None  # closed PRs are not shown


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _counts(merged: int, review: int) -> str:
    parts = [f"{merged} merged"] if merged else []
    if review:
        parts.append(f"{review} in review")
    return ", ".join(parts)


def build(config: dict, feed: dict) -> tuple[str, str, list[str]]:
    """Return (summary sentence, projects markdown, warnings). Pure: no I/O."""
    warnings: list[str] = []
    landed: dict[str, str] = config.get("landed") or {}
    picks: dict[str, str] = config.get("picks") or {}

    # Upstream work only: PRs into my own repos are not patches to anyone's stack.
    me = feed["user"]["username"].lower()
    prs: dict[str, dict] = {}
    for pr in feed["prs"]:
        bucket = _bucket(pr["state"])
        if bucket and pr["repo"].split("/")[0].lower() != me:
            prs[pr["url"]] = {"repo": pr["repo"], "title": pr["title"], "bucket": bucket}
    if not any(pr["bucket"] == "review" or url not in landed for url, pr in prs.items()):
        # A feed that parses but lists nothing is a broken upstream, not a
        # fact to publish; fail like a fetch error so the last deploy stays.
        raise ValueError("PR feed has no open or merged upstream PRs")
    for url in landed:
        repo = "/".join(url.split("/")[3:5])
        prs[url] = {"repo": repo, "title": picks.get(url, url), "bucket": "merged"}

    owner: dict[str, tuple[int, str | None]] = {}
    projects = []
    for i, project in enumerate(config["projects"]):
        projects.append({**project, "repos": dict(project["repos"])})
        for repo, label in project["repos"].items():
            owner[repo.lower()] = (i, label)
    for pr in prs.values():
        if pr["repo"].lower() not in owner:
            owner[pr["repo"].lower()] = (len(projects), None)
            org = pr["repo"].split("/")[0]
            projects.append({
                "name": pr["repo"],
                "url": f"https://github.com/{pr['repo']}",
                "logo": f"https://github.com/{org}.png?size=128",
                "repos": {pr["repo"]: None},
                "unclaimed": True,
            })

    for url in picks:
        if url not in prs:
            warnings.append(f"pick not open or merged, skipped: {url}")

    blocks = []
    total = {"merged": 0, "review": 0}
    for i, project in enumerate(projects):
        mine = [(url, pr) for url, pr in prs.items() if owner[pr["repo"].lower()][0] == i]
        if not mine:
            continue
        count = {"merged": 0, "review": 0}
        for _, pr in mine:
            count[pr["bucket"]] += 1
            total[pr["bucket"]] += 1

        if project.get("unclaimed"):
            # Raw GitHub titles: escape so `<T>` or `[x]` stays text.
            shown = [(url, html.escape(pr["title"]).replace("[", "\\[").replace("]", "\\]")) for url, pr in mine]
        else:
            shown = [(url, title) for url, title in picks.items() if url in prs and owner[prs[url]["repo"].lower()][0] == i]

        used = {pr["repo"].lower() for _, pr in mine}
        labels = [label for repo, label in project["repos"].items() if label and repo.lower() in used]
        date = _counts(count["merged"], count["review"])
        if labels:
            unlabeled = any(label is None and repo.lower() in used for repo, label in project["repos"].items())
            date += (", including " if unlabeled else " across ") + _join(labels)

        lines = [
            '<div class="pb-tl-item" markdown>',
            f'<div class="pb-tl-icon"><img src="{project["logo"]}" alt="" width="64" height="64" /></div>',
            '<div class="pb-tl-body" markdown>',
            f'<p class="pb-tl-title"><a href="{project["url"]}">{project["name"]}</a></p>',
            f'<p class="pb-tl-date">{date}</p>',
            "",
        ]
        for bucket, heading in (("merged", "Merged"), ("review", "In review")):
            items = [(url, title) for url, title in shown if prs[url]["bucket"] == bucket]
            if not items:
                continue
            lines += [f'<p class="pb-patch-group">{heading}</p>', ""]
            for url, title in items:
                label = owner[prs[url]["repo"].lower()][1]
                line = f"- [{title}]({url})"
                if url in landed:
                    line += f", {landed[url]}"
                if label:
                    line += f" ({label})"
                lines.append(line)
            lines.append("")
        lines += ["</div>", "</div>"]
        blocks.append("\n".join(lines))

    day = feed["generatedAt"][:10]
    summary = f"**{total['merged']} merged and {total['review']} in review** as of {day}"
    body = '<div class="pb-timeline pb-patches" markdown>\n\n' + "\n\n".join(blocks) + "\n\n</div>"
    return summary, body, warnings


def on_page_markdown(markdown: str, page, **_):
    if SUMMARY_MARK not in markdown and PROJECTS_MARK not in markdown:
        return markdown
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary, body, warnings = build(config, fetch(config["source"]))
    for warning in warnings:
        log.warning(warning)
    return markdown.replace(SUMMARY_MARK, summary).replace(PROJECTS_MARK, body)
