"""Keep the existing notebooks executable and connected to the published lessons."""

import ast
import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('foundations.ipynb', 'cnn_internals.ipynb', 'main.ipynb')


class NotebookGuideTests(unittest.TestCase):
    def test_perceptron_figure_keeps_values_legible_inside_nodes(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib import pyplot as plt
        from cifar10_lab.foundation_visualization import plot_perceptron_anatomy

        figure = plot_perceptron_anatomy()
        try:
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            renderer = canvas.get_renderer()
            axis = figure.axes[0]
            nodes = {artist.get_gid().removeprefix('perceptron-node-'): artist
                     for artist in axis.patches if (artist.get_gid() or '').startswith('perceptron-node-')}
            for text in axis.texts:
                identity = text.get_gid() or ''
                if identity.startswith('perceptron-label-'):
                    bounds = text.get_window_extent(renderer)
                    node_bounds = nodes[identity.removeprefix('perceptron-label-')].get_window_extent(renderer)
                    self.assertTrue(node_bounds.contains(bounds.x0, bounds.y0))
                    self.assertTrue(node_bounds.contains(bounds.x1, bounds.y1))
            labels = [text.get_text() for text in axis.texts]
            self.assertIn('x1\n1.00', labels)
            self.assertIn('x2\n0.50', labels)
            self.assertTrue(any('z = Σxw + b = +0.70' in text for text in labels))
            for text in axis.texts:
                if 'x·w=' in text.get_text():
                    self.assertIsNotNone(text.get_bbox_patch())
        finally:
            plt.close(figure)

    def test_all_python_cells_parse_without_rewriting_notebook_outputs(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    with self.subTest(notebook=name, cell=index):
                        ast.parse(''.join(cell['source']))

    def test_first_code_cell_reports_the_actual_kernel_before_optional_imports(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
            first = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
            tree = ast.parse(''.join(first['source']))
            executable_reports = [index for index, statement in enumerate(tree.body)
                                  if isinstance(statement, ast.Expr)
                                  and isinstance(statement.value, ast.Call)
                                  and any(isinstance(node, ast.Attribute)
                                          and isinstance(node.value, ast.Name)
                                          and node.value.id == 'sys'
                                          and node.attr == 'executable'
                                          for node in ast.walk(statement))]
            self.assertEqual(len(executable_reports), 1, name)
            optional_imports = [i for i, statement in enumerate(tree.body)
                                if isinstance(statement, (ast.Import, ast.ImportFrom))
                                and 'torch' in ast.unparse(statement)]
            self.assertLess(executable_reports[0], min(optional_imports), name)

    def test_notebook_web_links_resolve_to_existing_published_pages(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
            markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                                 if cell['cell_type'] == 'markdown')
            references = re.findall(r'https://tember40\.github\.io/Backbone_tester_CNN_Transformer/(docs/[^)\s]+)', markdown)
            self.assertTrue(references, name)
            for reference in references:
                target = ROOT / urlparse(reference).path
                self.assertTrue(target.is_file(), f'{name}: {reference}')


if __name__ == '__main__':
    unittest.main()
