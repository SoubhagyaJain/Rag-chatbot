"""Refresh docs/gen-interview-notes.html — metrics, responsive CSS/JS."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from _update_interview_notes_root import patch_css  # noqa: E402

GEN = ROOT / "company_policy_rag" / "docs" / "gen-interview-notes.html"
GEN_PY = ROOT / "company_policy_rag" / "docs" / "_gen_interview_notes.py"

NEW_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="theme-color" content="#e4eaf4">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <title>Rag-chatbot — Generator Docs · Interview Notes</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
"""

GEN_EXTRA_CSS = """
    .gen-hero-stats {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 0.65rem;
      margin-top: 1.25rem;
    }

    @media (max-width: 900px) {
      .gen-hero-stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .file-grid { grid-template-columns: 1fr 1fr; }
      .step-row { gap: 0.35rem; }
    }

    @media (max-width: 480px) {
      .file-grid { grid-template-columns: 1fr; }
      .src-cat { margin-left: 0; width: 100%; order: 3; }
      .source-card summary { gap: 0.35rem; }
      .step-row { flex-direction: column; align-items: stretch; }
      .step-arrow { display: none; }
      .step-chip { text-align: center; }
      .code-panel { font-size: 0.72rem; }
    }
"""

ENHANCED_JS = r"""
  <button type="button" class="back-to-top" id="back-to-top" aria-label="Back to top" title="Back to top">↑</button>

  <script>
    (function () {
      var toggle = document.getElementById('menu-toggle');
      var closeBtn = document.getElementById('toc-close');
      var panel = document.getElementById('toc-panel');
      var backdrop = document.getElementById('toc-backdrop');
      var backToTop = document.getElementById('back-to-top');
      var links = panel ? panel.querySelectorAll('nav a[href^="#"]') : [];
      var mobileMq = window.matchMedia('(max-width: 900px)');

      function isMobile() { return mobileMq.matches; }

      function openMenu() {
        if (!panel) return;
        panel.classList.add('open');
        backdrop.classList.add('visible');
        backdrop.setAttribute('aria-hidden', 'false');
        document.body.classList.add('menu-open');
        if (toggle) {
          toggle.setAttribute('aria-expanded', 'true');
          toggle.setAttribute('aria-label', 'Close navigation');
        }
      }

      function closeMenu() {
        if (!panel) return;
        panel.classList.remove('open');
        backdrop.classList.remove('visible');
        backdrop.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('menu-open');
        if (toggle) {
          toggle.setAttribute('aria-expanded', 'false');
          toggle.setAttribute('aria-label', 'Open navigation');
        }
      }

      if (toggle) {
        toggle.addEventListener('click', function () {
          panel.classList.contains('open') ? closeMenu() : openMenu();
        });
      }
      if (closeBtn) closeBtn.addEventListener('click', closeMenu);
      if (backdrop) backdrop.addEventListener('click', closeMenu);

      links.forEach(function (link) {
        link.addEventListener('click', function () {
          if (isMobile()) closeMenu();
        });
      });

      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && panel && panel.classList.contains('open')) closeMenu();
      });

      function onBreakpointChange() {
        if (!isMobile() && panel && panel.classList.contains('open')) closeMenu();
      }
      if (mobileMq.addEventListener) mobileMq.addEventListener('change', onBreakpointChange);
      else if (mobileMq.addListener) mobileMq.addListener(onBreakpointChange);
      window.addEventListener('resize', onBreakpointChange);

      function refreshScrollableTables() {
        document.querySelectorAll('.table-wrap').forEach(function (wrap) {
          var scrollable = wrap.scrollWidth > wrap.clientWidth + 2;
          wrap.classList.toggle('is-scrollable', scrollable);
        });
      }
      refreshScrollableTables();
      window.addEventListener('resize', refreshScrollableTables);
      window.addEventListener('load', refreshScrollableTables);

      var sections = [];
      links.forEach(function (link) {
        var id = link.getAttribute('href');
        if (!id || id.charAt(0) !== '#') return;
        var el = document.querySelector(id);
        if (el) sections.push({ link: link, el: el });
      });

      function updateActiveLink() {
        var marker = isMobile() ? 120 : 80;
        var current = sections[0] || null;
        sections.forEach(function (s) {
          if (s.el.getBoundingClientRect().top <= marker) current = s;
        });
        sections.forEach(function (s) {
          s.link.classList.toggle('active', s === current);
        });
      }
      if (sections.length) {
        window.addEventListener('scroll', updateActiveLink, { passive: true });
        window.addEventListener('resize', updateActiveLink);
        updateActiveLink();
      }

      if (backToTop) {
        backToTop.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        window.addEventListener('scroll', function () {
          backToTop.classList.toggle('visible', window.scrollY > 480);
        }, { passive: true });
      }

      var expandSrc = document.getElementById('expand-src');
      if (expandSrc) expandSrc.addEventListener('click', function () {
        document.querySelectorAll('.source-card').forEach(function (d) { d.open = true; });
      });
      var collapseSrc = document.getElementById('collapse-src');
      if (collapseSrc) collapseSrc.addEventListener('click', function () {
        document.querySelectorAll('.source-card').forEach(function (d) { d.open = false; });
      });
    })();
  </script>
"""

METRIC_REPLACEMENTS = [
    ("and 180 tests", "and 222 tests"),
    ("182 pytest tests", "222 pytest tests"),
    ("Guidebook: full rel 0.700 (run 164848, gate passed)", "Guidebook: rel 0.766, hit 0.886 (run 101844, exceeds gate)"),
    ("Run 164848: full guidebook rel 0.700 (gate passed)", "Run 101844: guidebook rel 0.766, hit 0.886 (best)"),
    ("guidebook faith 0.629 baseline", "guidebook faith 0.594 on run 101844"),
    ("640 tokens, 64 overlap", "hierarchical 2000/480 tokens"),
    (
        "Phase 4 CI green on GitHub (run 27804469869): pytest + ci_eval_gate.py retrieval smoke.",
        "Phase 4 CI + Docker CD green (runs 27804469869, 27820859129): 222 pytest + ci_eval_gate.py smoke; image soubhagya007/rag-chatbot.",
    ),
    (
        "Faithfulness tuning dd40b86 (055058): faith 0.543 vs baseline 0.629; next: code/currency retrieval.",
        "Faithfulness tuning dd40b86 (055058): faith 0.543 vs 0.594 baseline (101844); next: 5 weak code/abstention cases.",
    ),
    (
        "49 questions · 8 categories · Anthropic solo AI/ML engineer focus",
        "49 questions · 222 tests · run 101844 rel 0.766 · Docker CD green",
    ),
    (
        '<div class="file-pill"><strong>CSS source</strong><code>project-plans.html</code></div>',
        '<div class="file-pill"><strong>CSS source</strong><code>../../project-plans.html</code></div>',
    ),
    (
        "Python source → premium HTML/CSS · 49 Anthropic-weighted questions · liquid glass theme from project-plans.",
        "Python source → premium HTML/CSS · 49 Anthropic-weighted questions · synced with repo-root interview-notes.html · liquid glass theme from project-plans.",
    ),
    (
        "Re-reads base CSS from <code>project-plans.html</code> automatically",
        "Re-reads base CSS from repo-root <code>project-plans.html</code> automatically",
    ),
    (
        "Overwrites <code>interview-notes.html</code> and <code>gen-interview-notes.html</code>",
        "Writes repo-root <code>../../interview-notes.html</code> and <code>docs/gen-interview-notes.html</code>",
    ),
    (
        '<text class="node-label" x="812" y="58" text-anchor="middle">interview-notes.html</text>',
        '<text class="node-label" x="812" y="52" text-anchor="middle">interview-notes</text><text class="node-sub" x="812" y="68" text-anchor="middle">(repo root)</text>',
    ),
]

GEN_PY_REPLACEMENTS = METRIC_REPLACEMENTS + [
    ("640/64 tokens", "hierarchical 2000/480 tokens"),
    ("<strong>180</strong> Tests", "<strong>222</strong> Tests"),
    ("<strong>0.700</strong> Guidebook Rel", '<strong>0.766</strong> Guidebook Rel'),
    ("182 pytest across", "222 pytest across"),
    ("enumeration (164848 rel 0.700)", "topic pipelines (101844 rel 0.766)"),
    ("guidebook rel gate 0.700 (164848)", "guidebook rel 0.766 (101844)"),
    ("0.629→0.700 on full 35-case run", "0.629→0.766 on full 35-case run (101844)"),
    ("guidebook 0.629 today", "guidebook 0.594 on run 101844"),
    ("code-query rel 0.525 (run 164848)", "code/currency cases still weak (run 101844)"),
]


def patch_gen_html(html: str) -> str:
    css_match = re.search(r"<style>\s*(.*?)\s*</style>", html, re.DOTALL)
    if not css_match:
        raise RuntimeError("CSS block not found")
    css = patch_css(css_match.group(1))
    if ".gen-hero-stats" not in css:
        css = css.rstrip() + GEN_EXTRA_CSS
    html = NEW_HEAD + css + "\n  </style>\n</head>\n" + html.split("</head>", 1)[1]

    hero_stats = """
        <div class="gen-hero-stats hero-stats">
          <span class="stat-pill metric"><strong>0.747</strong> Policy Rel</span>
          <span class="stat-pill metric success"><strong>0.766</strong> Guidebook Rel</span>
          <span class="stat-pill metric success"><strong>0.886</strong> Hit Rate</span>
          <span class="stat-pill"><strong>222</strong> Tests</span>
          <span class="stat-pill success"><strong>CD</strong> Docker Green</span>
          <span class="stat-pill"><strong>49</strong> Questions</span>
        </div>"""
    if "gen-hero-stats" not in html:
        html = html.replace(
            """          <div class="file-pill"><strong>CSS source</strong><code>../../project-plans.html</code></div>
        </div>
      </header>

      <section id="g2">""",
            f"""          <div class="file-pill"><strong>CSS source</strong><code>../../project-plans.html</code></div>
        </div>{hero_stats}
      </header>

      <section id="g2">""",
        )

    html = html.replace(
        '<button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation">',
        '<button class="menu-toggle" id="menu-toggle" type="button" aria-label="Open navigation" aria-expanded="false" aria-controls="toc-panel">',
    )

    for old, new in METRIC_REPLACEMENTS:
        html = html.replace(old, new)

    html = html.replace(
        """      <footer>
        <p><strong>Company Policy RAG</strong> — Generator Documentation<br>
        <a href="../../interview-notes.html">Interview Notes</a> · <a href="../../project-plans.html">Engineering Plans</a> · <a href="https://github.com/SoubhagyaJain/Rag-chatbot">GitHub</a></p>
      </footer>""",
        """      <footer>
        <p>
          <strong>Rag-chatbot</strong> — Interview Notes Generator Docs · Updated 2026-06-19<br>
          <a href="../../README.md">Repo README</a> ·
          <a href="../../interview-notes.html">Interview Notes</a> ·
          <a href="../../project-plans.html">Engineering Plans</a> ·
          <a href="../../company_policy_rag/README3.md">README3</a> ·
          Docker Hub: <a href="https://hub.docker.com/r/soubhagya007/rag-chatbot" target="_blank" rel="noopener">soubhagya007/rag-chatbot</a> ·
          GitHub: <a href="https://github.com/SoubhagyaJain/Rag-chatbot" target="_blank" rel="noopener">SoubhagyaJain/Rag-chatbot</a>
        </p>
      </footer>""",
    )

    html = re.sub(r"\s*<script>.*?</script>\s*</body>", ENHANCED_JS + "\n</body>", html, flags=re.DOTALL)
    return html


def patch_gen_py(text: str) -> str:
    for old, new in GEN_PY_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main() -> None:
    html = GEN.read_text(encoding="utf-8")
    GEN.write_text(patch_gen_html(html), encoding="utf-8")
    print(f"Updated {GEN}")

    if GEN_PY.exists():
        py = GEN_PY.read_text(encoding="utf-8")
        GEN_PY.write_text(patch_gen_py(py), encoding="utf-8")
        print(f"Updated {GEN_PY}")


if __name__ == "__main__":
    main()