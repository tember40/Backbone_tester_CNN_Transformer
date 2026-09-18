import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.scripts = []
        self.stylesheets = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if (
            tag == "link"
            and attributes.get("rel") == "stylesheet"
            and attributes.get("href")
        ):
            self.stylesheets.append(attributes["href"])


def parse_page(path):
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def local_target(page, reference):
    parsed = urlparse(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("mailto:"):
        return None
    if not parsed.path:
        return page
    return (page.parent / unquote(parsed.path)).resolve()


class StaticSiteTests(unittest.TestCase):
    def test_internal_links_and_assets_exist(self):
        html_pages = sorted(DOCS.rglob("*.html"))
        self.assertGreaterEqual(len(html_pages), 2)

        for page in html_pages:
            parser = parse_page(page)
            references = parser.links + parser.scripts + parser.stylesheets
            for reference in references:
                target = local_target(page, reference)
                if target is None:
                    continue
                self.assertTrue(target.exists(), f"{page}: missing {reference}")

                fragment = urlparse(reference).fragment
                if fragment and target.suffix == ".html":
                    self.assertIn(
                        fragment,
                        parse_page(target).ids,
                        f"{page}: missing fragment {reference}",
                    )

    def test_perceptron_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "01-perceptron.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "anatomy",
            "calculator",
            "code",
            "learning",
            "limits",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "cifar10_lab" / "foundations.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "return features @ self.weights + self.bias",
            "error = target_value - prediction",
            "self.weights += self.learning_rate * error * sample",
            "self.bias += self.learning_rate * error",
            "return self.history",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

    def test_repository_page_redirects_to_learning_site(self):
        entry = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertIn('content="0; url=docs/"', entry)
        self.assertIn('window.location.replace("docs/"', entry)

        chapter_entry = (ROOT / "chapters" / "01-perceptron.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("../docs/chapters/01-perceptron.html", chapter_entry)
        self.assertIn("target.hash = window.location.hash", chapter_entry)

        second_chapter_entry = (
            ROOT / "chapters" / "02-linear-separability.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "../docs/chapters/02-linear-separability.html", second_chapter_entry
        )
        self.assertIn("target.hash = window.location.hash", second_chapter_entry)

        third_chapter_entry = (
            ROOT / "chapters" / "03-multilayer-perceptron.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "../docs/chapters/03-multilayer-perceptron.html", third_chapter_entry
        )
        self.assertIn("target.hash = window.location.hash", third_chapter_entry)

    def test_linear_separability_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "02-linear-separability.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "geometry",
            "xor",
            "code",
            "lab",
            "bridge",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "cifar10_lab" / "foundations.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "scores = feature_tensor @ weight_tensor + float(bias)",
            "predictions = (scores >= 0).to(torch.long)",
            "correct = predictions.eq(target_tensor)",
            "accuracy = correct.float().mean().item()",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line.replace(">", "&gt;"), source)

    def test_reusable_chapter_template_exists(self):
        template = DOCS / "templates" / "chapter-template.html"
        source = template.read_text(encoding="utf-8")
        for component in (
            "learning-objectives",
            "equation-card",
            "callout-question",
            "chapter-summary-box",
        ):
            self.assertIn(component, source)

    def test_multilayer_perceptron_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "03-multilayer-perceptron.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "anatomy",
            "forward",
            "learning",
            "code",
            "lab",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "cifar10_lab" / "foundations.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "hidden_linear = self.hidden(features)",
            "hidden = self.activation(hidden_linear)",
            "logits = self.output(hidden).squeeze(-1)",
            "probabilities = torch.sigmoid(logits)",
            'return self.forward_trace(features)["logits"]',
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

    def test_kwangwoon_visual_tokens_and_background_exist(self):
        stylesheet = (DOCS / "assets" / "css" / "site.css").read_text(
            encoding="utf-8"
        )
        for color in ("#7c192d", "#f7f6f3", "#fbf4ef", "#c6bfaa", "#fbae40"):
            self.assertIn(color, stylesheet.lower())

        monoline = DOCS / "assets" / "img" / "kw-monoline.svg"
        self.assertTrue(monoline.exists())
        self.assertIn("#C6BFAA", monoline.read_text(encoding="utf-8"))

    def test_chapter_navigation_stays_visible_and_layout_is_balanced(self):
        stylesheet = (DOCS / "assets" / "css" / "site.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".site-header {\n  position: fixed;", stylesheet)
        self.assertIn(".chapter-sidebar { position: fixed;", stylesheet)
        self.assertIn(
            "grid-template-columns: var(--sidebar) minmax(0, 1fr) var(--sidebar)",
            stylesheet,
        )
        self.assertIn(
            ".chapter-page .site-footer { width: calc(100% - (var(--sidebar) * 2)); margin-left: var(--sidebar); }",
            stylesheet,
        )

        for page in (DOCS / "chapters").glob("*.html"):
            source = page.read_text(encoding="utf-8")
            self.assertNotIn("sidebar-note", source, str(page))
            self.assertNotIn("이 장의 분량", source, str(page))

    def test_user_facing_pages_avoid_book_authorship_wording(self):
        for page in DOCS.rglob("*.html"):
            self.assertNotIn("교재", page.read_text(encoding="utf-8"), str(page))


if __name__ == "__main__":
    unittest.main()
