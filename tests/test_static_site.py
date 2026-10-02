import unittest
import re
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
        self.images = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])
        if tag == "script" and attributes.get("src"):
            self.scripts.append(attributes["src"])
        if tag == "img" and attributes.get("src"):
            self.images.append(attributes["src"])
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
            references = parser.links + parser.scripts + parser.stylesheets + parser.images
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

        fourth_chapter_entry = (
            ROOT / "chapters" / "04-convolution-filters.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "../docs/chapters/04-convolution-filters.html", fourth_chapter_entry
        )
        self.assertIn("target.hash = window.location.hash", fourth_chapter_entry)

        fifth_chapter_entry = (
            ROOT / "chapters" / "05-feature-maps-activations.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "../docs/chapters/05-feature-maps-activations.html",
            fifth_chapter_entry,
        )
        self.assertIn("target.hash = window.location.hash", fifth_chapter_entry)

        sixth_chapter_entry = (
            ROOT / "chapters" / "06-pooling-receptive-field.html"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "../docs/chapters/06-pooling-receptive-field.html",
            sixth_chapter_entry,
        )
        self.assertIn("target.hash = window.location.hash", sixth_chapter_entry)

        seventh_chapter_entry = (
            ROOT / "chapters" / "07-alexnet.html"
        ).read_text(encoding="utf-8")
        self.assertIn("../docs/chapters/07-alexnet.html", seventh_chapter_entry)
        self.assertIn("target.hash = window.location.hash", seventh_chapter_entry)

        eighth_chapter_entry = (
            ROOT / "chapters" / "08-vgg.html"
        ).read_text(encoding="utf-8")
        self.assertIn("../docs/chapters/08-vgg.html", eighth_chapter_entry)
        self.assertIn("target.hash = window.location.hash", eighth_chapter_entry)

        ninth_chapter_entry = (
            ROOT / "chapters" / "09-resnet.html"
        ).read_text(encoding="utf-8")
        self.assertIn("../docs/chapters/09-resnet.html", ninth_chapter_entry)
        self.assertIn("target.hash = window.location.hash", ninth_chapter_entry)

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

    def test_convolution_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "04-convolution-filters.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "principles",
            "calculation",
            "geometry",
            "code",
            "lab",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (
            ROOT / "cifar10_lab" / "cnn_visualization.py"
        ).read_text(encoding="utf-8")
        for code_line in (
            "image_row = output_row * stride",
            "image_column = output_column * stride",
            "products = patch * kernel_tensor",
            "value = products.sum()",
            "output[output_row, output_column] = value",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

    def test_chapter_navigation_stays_visible_and_layout_is_balanced(self):
        stylesheet = (DOCS / "assets" / "css" / "site.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".site-header {\n  position: fixed;", stylesheet)
        self.assertIn(".chapter-sidebar { position: fixed;", stylesheet)
        self.assertIn(".chapter-glossary { position: fixed;", stylesheet)
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

    def test_all_chapters_have_linked_glossaries(self):
        glossary = (DOCS / "assets" / "js" / "glossary.js").read_text(
            encoding="utf-8"
        )
        for chapter in range(1, 12):
            number = f"{chapter:02d}"
            page = next((DOCS / "chapters").glob(f"{number}-*.html"))
            source = page.read_text(encoding="utf-8")
            self.assertIn('id="chapterGlossary"', source)
            self.assertIn(f'data-chapter="{number}"', source)
            self.assertIn('../assets/js/glossary.js?', source)
            self.assertIn('class="glossary-list"', source)
            self.assertIn(f'"{number}": [[', glossary)

        definitions = {
            key: (chapter, section)
            for key, chapter, section in re.findall(
                r'^\s*(\w+): term\(.+, "(\d\d)", "([^"]+)"\),?$',
                glossary,
                re.MULTILINE,
            )
        }
        self.assertGreaterEqual(len(definitions), 30)
        for chapter in range(1, 12):
            number = f"{chapter:02d}"
            page = next((DOCS / "chapters").glob(f"{number}-*.html"))
            match = re.search(
                rf'^\s*"{number}": \[(.*)\],?$', glossary, re.MULTILINE
            )
            self.assertIsNotNone(match)
            entries = re.findall(r'\["(\w+)", "([^"]+)"\]', match.group(1))
            self.assertGreaterEqual(len(entries), 5)
            for key, local_section in entries:
                self.assertIn(key, definitions)
                self.assertIn(local_section, parse_page(page).ids)
                original_chapter, original_section = definitions[key]
                original_page = next(
                    (DOCS / "chapters").glob(f"{original_chapter}-*.html")
                )
                self.assertIn(original_section, parse_page(original_page).ids)

    def test_feature_maps_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "05-feature-maps-activations.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "tensor",
            "channels",
            "activation",
            "code",
            "lab",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (
            ROOT / "cifar10_lab" / "cnn_visualization.py"
        ).read_text(encoding="utf-8")
        for code_line in (
            '"shape": tuple(feature_tensor.shape)',
            '"minimum": float(feature_tensor.min().item())',
            '"maximum": float(feature_tensor.max().item())',
            '"positive_ratio": float((feature_tensor > 0).float().mean().item())',
            '"zero_ratio": float((feature_tensor == 0).float().mean().item())',
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line.replace(">", "&gt;"), source)

    def test_pooling_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "06-pooling-receptive-field.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "pooling",
            "geometry",
            "comparison",
            "code",
            "receptive-field",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (
            ROOT / "cifar10_lab" / "cnn_visualization.py"
        ).read_text(encoding="utf-8")
        for code_line in (
            "input_row = output_row * stride",
            "input_column = output_column * stride",
            "window = feature_tensor[",
            'value = window.max() if mode == "max" else window.mean()',
            "output[output_row, output_column] = value",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

    def test_alexnet_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "07-alexnet.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "breakthrough",
            "architecture",
            "adaptation",
            "code",
            "lab",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "backbone" / "AlexNet.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "x = self.features(x)",
            "x = self.avgpool(x)",
            "x = torch.flatten(x, 1)",
            "x = self.classifier(x)",
            "return x",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

    def test_vgg_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "08-vgg.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview",
            "paper",
            "small-kernels",
            "architecture",
            "adaptation",
            "code",
            "lab",
            "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "backbone" / "VGG.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "if v == 'M':",
            "layers += [nn.MaxPool2d(kernel_size=2, stride=2)]",
            "conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)",
            "layers += [conv2d, nn.ReLU(True)]",
            "in_channels = v",
            "return nn.Sequential(*layers)",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

        curriculum = (DOCS / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="chapters/08-vgg.html"', curriculum)
        alexnet = (DOCS / "chapters" / "07-alexnet.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('href="08-vgg.html"', alexnet)

    def test_resnet_page_contains_the_lesson_contract(self):
        page = DOCS / "chapters" / "09-resnet.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview", "paper", "residual", "blocks", "architecture",
            "adaptation", "code", "extensions", "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "backbone" / "ResNet.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "identity = x",
            "out = self.conv1(x)",
            "out = self.bn2(out)",
            "identity = self.downsample(x)",
            "out += identity",
            "out = self.relu(out)",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)

        self.assertIn('href="09-resnet.html"', (
            DOCS / "chapters" / "08-vgg.html"
        ).read_text(encoding="utf-8"))
        self.assertIn('href="chapters/09-resnet.html"', (
            DOCS / "index.html"
        ).read_text(encoding="utf-8"))
        self.assertIn("7×7 합성곱, 64채널", source)
        self.assertIn("32×32 RGB", source)

    def test_home_page_timeline_and_private_curriculum_states(self):
        home = (DOCS / "index.html").read_text(encoding="utf-8")
        self.assertEqual(home.count('data-year="'), 9)
        self.assertEqual(home.count('class="timeline-step'), 9)
        self.assertIn('src="assets/js/home.js?', home)
        self.assertIn('id="timelineDescription"', home)
        self.assertIn('id="timelineSource"', home)
        self.assertIn('id="timelineChapter"', home)
        self.assertNotIn("학습 가능", home)
        self.assertEqual(home.count('class="is-locked"'), 6)
        self.assertEqual(home.count('class="private-label">비공개'), 6)
        self.assertNotIn("lock-icon", home)
        self.assertIn("아직 제작 중인 단원", home)

        script = (DOCS / "assets" / "js" / "home.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('card.addEventListener("pointerenter"', script)
        self.assertIn('card.addEventListener("focus"', script)
        self.assertIn('closest(".timeline-step")', script)

    def test_senet_chapter_and_study_reminder(self):
        page = DOCS / "chapters" / "10-senet.html"
        source = page.read_text(encoding="utf-8")
        for section_id in (
            "overview", "paper", "squeeze", "excitation", "reduction",
            "placement", "adaptation", "code", "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "backbone" / "SeNet.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "b, c, _, _ = x.size()",
            "y = self.avg_pool(x).view(b, c)",
            "y = self.fc(y).view(b, c, 1, 1)",
            "return x * y.expand_as(x)",
            "out = self.se(out)",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)
        self.assertIn('href="chapters/10-senet.html"', (
            DOCS / "index.html"
        ).read_text(encoding="utf-8"))
        self.assertIn('href="10-senet.html"', (
            DOCS / "chapters" / "09-resnet.html"
        ).read_text(encoding="utf-8"))

        reminder = "개념과 결과는 원 논문, 공식 문서, 실제 코드와 함께 확인해 주세요."
        for public_page in [DOCS / "index.html", *(DOCS / "chapters").glob("*.html")]:
            self.assertIn(reminder, public_page.read_text(encoding="utf-8"))

    def test_mobilenetv2_chapter_matches_implementation(self):
        page = DOCS / "chapters" / "11-mobilenetv2.html"
        source = page.read_text(encoding="utf-8")
        self.assertIn('src="../assets/img/mobilenet-depthwise-pointwise.svg?v=20261001b"', source)
        self.assertIn("MobileNet 원 논문의 Figure 2", source)
        for section_id in (
            "overview", "paper", "depthwise", "inverted", "linear",
            "architecture", "code", "adaptation", "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)

        implementation = (ROOT / "backbone" / "MobileNet.py").read_text(
            encoding="utf-8"
        )
        for code_line in (
            "hidden_dim = int(round(inp * expand_ratio))",
            "self.use_res_connect = self.stride == 1 and inp == oup",
            "groups=hidden_dim",
            "nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)",
            "return x + self.conv(x)",
            "x.mean([2, 3])",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)
        self.assertIn('href="chapters/11-mobilenetv2.html"', (
            DOCS / "index.html"
        ).read_text(encoding="utf-8"))
        self.assertIn('href="11-mobilenetv2.html"', (
            DOCS / "chapters" / "10-senet.html"
        ).read_text(encoding="utf-8"))

    def test_efficientnet_chapter_matches_implementation(self):
        source = (DOCS / "chapters" / "12-efficientnet.html").read_text(encoding="utf-8")
        for section_id in (
            "overview", "paper", "dimensions", "compound", "mbconv",
            "presets", "code", "adaptation", "checkpoint",
        ):
            self.assertIn(f'id="{section_id}"', source)
        implementation = (ROOT / "backbone" / "EfficientNet.py").read_text(encoding="utf-8")
        for code_line in (
            "output_channel = _make_divisible(c * width_mult, round_nearest)",
            "num_blocks = int(math.ceil(n * depth_mult))",
            "stride = s if i == 0 else 1",
            "features.append(block(input_channel, output_channel, stride, expand_ratio=t, kernel_size=k, skip_connection=True, norm_layer=norm_layer, se_layer=se_layer))",
        ):
            self.assertIn(code_line, implementation)
            self.assertIn(code_line, source)
        self.assertIn('href="chapters/12-efficientnet.html"', (DOCS / "index.html").read_text(encoding="utf-8"))
        self.assertIn('href="12-efficientnet.html"', (DOCS / "chapters" / "11-mobilenetv2.html").read_text(encoding="utf-8"))
        glossary = (DOCS / "assets" / "js" / "glossary.js").read_text(encoding="utf-8")
        self.assertIn('"12": "12-efficientnet.html"', glossary)

    def test_user_facing_pages_avoid_book_authorship_wording(self):
        for page in DOCS.rglob("*.html"):
            self.assertNotIn("교재", page.read_text(encoding="utf-8"), str(page))


if __name__ == "__main__":
    unittest.main()
