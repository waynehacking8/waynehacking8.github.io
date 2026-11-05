import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
from patches import build  # noqa: E402

CONFIG = {
    "projects": [
        {"name": "vLLM", "url": "u", "logo": "l", "repos": {"vllm-project/vllm": None, "vllm-project/llm-compressor": "llm-compressor"}},
        {"name": "NVIDIA", "url": "u", "logo": "l", "repos": {"NVIDIA/cutlass": "CUTLASS"}},
        {"name": "Empty", "url": "u", "logo": "l", "repos": {"x/y": None}},
    ],
    "landed": {"https://github.com/pytorch/pytorch/pull/1": "landed as `abc`"},
    "picks": {
        "https://github.com/vllm-project/vllm/pull/2": "Short two",
        "https://github.com/vllm-project/vllm/pull/1": "Short one",
        "https://github.com/vllm-project/llm-compressor/pull/3": "Compressor",
        "https://github.com/NVIDIA/cutlass/pull/4": "Cutlass",
        "https://github.com/vllm-project/vllm/pull/9": "Now closed",
        "https://github.com/pytorch/pytorch/pull/1": "Landed fix",
    },
}


def pr(repo, n, state):
    return {"repo": repo, "url": f"https://github.com/{repo}/pull/{n}", "title": f"Raw {n}", "state": state}


FEED = {
    "generatedAt": "2026-11-05T13:13:14.536Z",
    "user": {"username": "Me"},
    "prs": [
        pr("me/own-lab", 8, "merged"),
        pr("vllm-project/vllm", 1, "merged"),
        pr("vllm-project/vllm", 2, "open"),
        pr("vllm-project/vllm", 5, "draft"),
        pr("vllm-project/vllm", 9, "closed"),
        pr("vllm-project/llm-compressor", 3, "merged"),
        pr("NVIDIA/cutlass", 4, "open"),
        pr("new-org/new-repo", 7, "open"),
    ],
}


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.summary, self.body, self.warnings = build(CONFIG, FEED)

    def section(self, name):
        start = self.body.index(f">{name}</a>")
        end = self.body.find('<div class="pb-tl-item"', start)
        return self.body[start:end if end != -1 else None]

    def test_summary_counts_every_pr_and_landed_ones(self):
        # merged: vllm#1, compressor#3, landed pytorch#1; review: vllm#2, #5 draft, cutlass#4, new#7
        self.assertEqual(self.summary, "**3 merged and 4 in review** as of 2026-11-05")

    def test_state_comes_from_the_feed_not_the_config(self):
        vllm = self.section("vLLM")
        merged, review = vllm.split("In review")
        self.assertIn("[Short one](https://github.com/vllm-project/vllm/pull/1)", merged)
        self.assertIn("[Short two](https://github.com/vllm-project/vllm/pull/2)", review)

    def test_project_line_counts_unpicked_prs_and_names_sub_repos(self):
        self.assertIn("2 merged, 2 in review, including llm-compressor", self.section("vLLM"))
        self.assertIn("1 in review across CUTLASS", self.section("NVIDIA"))
        self.assertIn("[Compressor](https://github.com/vllm-project/llm-compressor/pull/3) (llm-compressor)", self.section("vLLM"))

    def test_closed_pick_is_dropped_with_a_warning(self):
        self.assertNotIn("Now closed", self.body)
        self.assertEqual(self.warnings, ["pick not open or merged, skipped: https://github.com/vllm-project/vllm/pull/9"])

    def test_landed_pr_counts_as_merged_with_its_note(self):
        self.assertIn("[Landed fix](https://github.com/pytorch/pytorch/pull/1), landed as `abc`", self.body)

    def test_unclaimed_repo_gets_its_own_group_and_empty_projects_vanish(self):
        self.assertIn("[Raw 7](https://github.com/new-org/new-repo/pull/7)", self.section("new-org/new-repo"))
        self.assertNotIn(">Empty</a>", self.body)

    def test_prs_into_my_own_repos_are_not_patches(self):
        self.assertNotIn("own-lab", self.body)


if __name__ == "__main__":
    unittest.main()
