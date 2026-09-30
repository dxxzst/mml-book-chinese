"""Bibliography regression tests; no PDF or third-party modules required."""

from pathlib import Path
import tempfile
import unittest

from build_references import extract_references, format_reference, write_references


TOUSSAINT_URL = (
    'https://ipvs.informatik.uni-stuttgart.de/mlr/marc/notes/gradientDescent.pdf'
)


def link(url):
    return f'[{url}](<{url}>)'


class ReferenceFormattingTests(unittest.TestCase):
    def test_original_wrapped_hostname(self):
        self.assertEqual(
            format_reference([
                'Toussaint, Marc. 2012. Some Notes on Gradient Descent. '
                'https://ipvs.informatik.uni-',
                'stuttgart.de/mlr/marc/notes/gradientDescent.pdf.',
            ]),
            'Toussaint, Marc. 2012. Some Notes on Gradient Descent. '
            + link(TOUSSAINT_URL) + '.',
        )

    def test_wrapped_hostname_preserves_hyphen_and_indentation(self):
        for scheme in ('http', 'https'):
            with self.subTest(scheme=scheme):
                url = f'{scheme}://www.uni-stuttgart.de/notes.pdf'
                self.assertEqual(
                    format_reference([
                        f'{scheme}://www.uni- \r',
                        '\tstuttgart.de/notes.pdf.',
                    ]),
                    link(url) + '.',
                )

    def test_prose_hyphens_are_unchanged(self):
        self.assertEqual(
            format_reference([
                'Practical Bayesian Op-',
                'timization. Wellesley-Cambridge Press. A uni-',
                'stuttgart.de reference.',
            ]),
            'Practical Bayesian Op- timization. Wellesley-Cambridge Press. '
            'A uni- stuttgart.de reference.',
        )

    def test_same_line_spaces_are_not_guessed_to_be_url_wraps(self):
        self.assertEqual(
            format_reference(['https://www.uni- stuttgart.de/notes.pdf.']),
            link('https://www.uni-') + ' stuttgart.de/notes.pdf.',
        )

    def test_complete_url_does_not_consume_next_prose_line(self):
        for url in ('https://example.org/', 'https://example.org/notes.pdf'):
            with self.subTest(url=url):
                self.assertEqual(
                    format_reference([url, 'Journal of Machine Learning.']),
                    link(url) + ' Journal of Machine Learning.',
                )

    def test_url_path_hyphen_is_not_treated_as_a_hostname_wrap(self):
        self.assertEqual(
            format_reference(['https://example.org/data-', 'model.org results.']),
            link('https://example.org/data-') + ' model.org results.',
        )

    def test_punctuation_stays_outside_link(self):
        self.assertEqual(
            format_reference(['See (https://example.org/notes.pdf).']),
            'See (' + link('https://example.org/notes.pdf') + ').',
        )

    def test_balanced_parentheses_belong_to_url(self):
        url = 'https://example.org/notes(v2).pdf'
        self.assertEqual(format_reference([url + '.']), link(url) + '.')

    def test_multiple_urls_and_accents(self):
        self.assertEqual(
            format_reference([
                'Sch¨olkopf. https://example.org/a,',
                'https://example.org/b.',
            ]),
            'Schölkopf. ' + link('https://example.org/a') + ', '
            + link('https://example.org/b') + '.',
        )


class ExtractionTests(unittest.TestCase):
    def test_join_survives_block_boundary_and_keeps_entry_numbering(self):
        class Page:
            def __init__(self, texts=()):
                self.texts = texts

            def get_text(self, mode):
                assert mode == 'blocks'
                return [(0, 0, 0, 0, text) for text in self.texts]

        doc = {pno: Page() for pno in range(400, 412)}
        doc[409] = Page([
            'References\n404\n'
            'Toussaint, Marc. 2012. Some Notes on Gradient Descent. '
            'https://ipvs.informatik.uni-\n',
            'stuttgart.de/mlr/marc/notes/gradientDescent.pdf.\n'
            'Trefethen, Lloyd N., and Bau III, David. 1997. '
            'Numerical Linear Algebra. SIAM.\n',
            'Draft (2019-10-27) of “Mathematics for Machine Learning”. '
            'Feedback: https://mml-book.com.\n',
        ])
        entries = extract_references(doc)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0],
                         'Toussaint, Marc. 2012. Some Notes on Gradient Descent. '
                         + link(TOUSSAINT_URL) + '.')
        self.assertEqual(entries[1],
                         'Trefethen, Lloyd N., and Bau III, David. 1997. '
                         'Numerical Linear Algebra. SIAM.')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'references.md'
            write_references(entries, output)
            self.assertEqual(output.read_text(encoding='utf-8'),
                             '# 参考文献 (References)\n\n'
                             '本书涉及的所有学术专著、经典论文及技术报告的完整参考文献列表如下：\n\n'
                             f'1. {entries[0]}\n\n2. {entries[1]}\n\n')


if __name__ == '__main__':
    unittest.main()
