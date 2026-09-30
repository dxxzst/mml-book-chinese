#!/usr/bin/env node
// Render complete Markdown with the Markdown Preview Enhanced core engine.
const fs = require('node:fs');
const path = require('node:path');
const { Notebook, wrapNodeFSAsApi } = require('crossnote');
const cheerio = require('cheerio');

const ROOT = path.resolve(__dirname, '..');
const BOOK_PATHS = [
  'README.md', '0.Foreword.md', '3.References.md', '4.Index.md',
  '1.Part I Mathematical Foundations',
  '2.Part II Central Machine Learning Problems',
];

function markdownFiles(paths) {
  return paths.flatMap(file => {
    if (fs.statSync(file).isDirectory()) {
      return markdownFiles(fs.readdirSync(file, { withFileTypes: true })
        .filter(entry => !entry.name.startsWith('.') && entry.name !== 'node_modules'
          && !entry.isSymbolicLink())
        .map(entry => path.join(file, entry.name)));
    }
    return file.endsWith('.md') ? [file] : [];
  }).sort();
}

// Source locations come from Markdown tokens, not a second regex math parser.
// Inline tokens (including displays in blockquotes) inherit the parent's map.
function addSourceLocations(md) {
  md.core.ruler.push('qa_math_locations', state => {
    for (const token of state.tokens) {
      if (token.type === 'math_block') {
        token.meta.qaLine = token.map[0] + 1;
      }
      if (token.type !== 'inline' || !token.map) continue;
      let cursor = 0;
      for (const child of token.children || []) {
        if (child.type !== 'math') continue;
        const start = token.content.indexOf(child.meta.openTag, cursor);
        const contentStart = token.content.indexOf(child.content,
          start < 0 ? cursor : start + child.meta.openTag.length);
        child.meta.qaLine = token.map[0] + 1
          + token.content.slice(0, Math.max(0, contentStart)).split('\n').length - 1;
        cursor = Math.max(cursor, contentStart + child.content.length
          + child.meta.closeTag.length);
      }
    }
  });
  for (const type of ['math', 'math_block']) {
    const render = md.renderer.rules[type];
    md.renderer.rules[type] = (tokens, index, ...args) => {
      const html = render(tokens, index, ...args);
      if (!html.includes('ParseError:') && !html.includes('katex-error')) return html;
      const token = tokens[index];
      const tag = token.content.match(/\\tag\*?\{([^}]+)\}/)?.[1] || '';
      return `<span data-qa-math-line="${token.meta.qaLine || 1}" `
        + `data-qa-math-tag="${md.utils.escapeHtml(tag)}">${html}</span>`;
    };
  }
}

async function createRenderer(root = ROOT) {
  // Do not evaluate notebook-local config/parser scripts or write generated files.
  const fileSystem = wrapNodeFSAsApi();
  const exists = fileSystem.exists;
  fileSystem.exists = file => path.resolve(file) === path.join(root, '.crossnote')
    ? Promise.resolve(false) : exists(file);
  for (const method of ['writeFile', 'mkdir', 'unlink']) {
    fileSystem[method] = async () => { throw new Error('QA rendering is read-only'); };
  }
  const notebook = await Notebook.init({ notebookPath: root, fs: fileSystem, config: {
    markdownParser: 'markdown-it',
    mathRenderingOption: 'KaTeX',
    mathInlineDelimiters: [['$', '$']],
    mathBlockDelimiters: [['$$', '$$']],
    enableScriptExecution: false,
    previewTheme: 'github-light.css',
    katexConfig: { throwOnError: true, trust: false, strict: 'warn' },
  } });
  addSourceLocations(notebook.md);
  return async (file, source) => {
    const output = await notebook.getNoteMarkdownEngine(file).parseMD(source, {
      isForPreview: true, useRelativeFilePath: true, hideFrontMatter: false,
      runAllCodeChunks: false, triggeredBySave: false,
    });
    const $ = cheerio.load(output.html);
    const errors = $('span[style], .katex-error').filter((_, element) =>
      $(element).hasClass('katex-error')
        || $(element).text().startsWith('ParseError: KaTeX parse error:'))
      .map((_, element) => {
        const node = $(element);
        const location = node.closest('[data-qa-math-line]');
        // Fenced math is rendered by a later Crossnote stage. Its preview
        // source-line attribute locates failures that bypass the math rules.
        const sourceNode = node.closest('[data-source-line]');
        return {
          line: Number(location.attr('data-qa-math-line')
            || sourceNode.attr('data-source-line') || 1),
          tag: location.attr('data-qa-math-tag') || '',
          message: node.attr('title') || node.text(),
        };
      }).get();
    return { file, renderedFormulas: $('.katex').length, errors };
  };
}

async function main(args = process.argv.slice(2)) {
  const files = markdownFiles(args.length ? args.map(file => path.resolve(file))
    : BOOK_PATHS.map(file => path.join(ROOT, file)));
  if (!files.length) throw new Error('No Markdown files selected');
  const render = await createRenderer();
  let formulas = 0;
  let errors = 0;
  for (const file of files) {
    const relative = path.relative(ROOT, file).split(path.sep).join('/');
    const result = await render(relative, fs.readFileSync(file, 'utf8'));
    formulas += result.renderedFormulas;
    errors += result.errors.length;
    for (const error of result.errors) {
      console.error(`${relative}:${error.line}: ${error.tag ? `(${error.tag}) ` : ''}`
        + error.message.replace(/\s+/g, ' '));
    }
  }
  console.log(`Math render: ${files.length} files, ${formulas} formulas, ${errors} errors `
    + `(Crossnote 0.9.41 / KaTeX ${require('katex').version}).`);
  return errors ? 1 : 0;
}

module.exports = { createRenderer, markdownFiles, main };
if (require.main === module) {
  main().then(code => { process.exitCode = code; }).catch(error => {
    console.error(`Math verification failed: ${error.message}`);
    process.exitCode = 2;
  });
}
