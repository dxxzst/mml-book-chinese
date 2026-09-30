const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { test } = require('node:test');
const { createRenderer } = require('./verify_math.cjs');

// The three broken book expressions as Markdown, including their original
// blockquote / separate-display context. Do not repair this negative fixture.
const broken = String.raw`# Regression examples

> Therefore:
> $$
> \begin{aligned}
> f^*(\boldsymbol \alpha) &= \frac{1}{\lambda} \boldsymbol \alpha^\top \boldsymbol K \boldsymbol \alpha - \frac{\lambda}{2} \left( \frac{1}{\lambda} \boldsymbol K \boldsymbol \alpha \right)^\top \boldsymbol K^{-1} \left( \frac{1}{\lambda} \boldsymbol K \boldsymbol \alpha \right) \\
> &= \frac{1}{2\lambda} \boldsymbol \alpha^\top \boldsymbol K \boldsymbol \alpha
> </aligned} \tag{7.62}
> $$

$$
= \pi_k (2\pi)^{-\frac{D}{2}} \left[ \frac{\partial}{\partial \boldsymbol{\Sigma}_k}\det(\boldsymbol{\Sigma}_k)^{-\frac{1}{2}} \exp\left( -\frac{1}{2}(\boldsymbol{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_n - \boldsymbol{\mu}_k) \right)
\tag{11.32b}
$$

$$
\qquad + \det(\boldsymbol{\Sigma}_k)^{-\frac{1}{2}} \frac{\partial}{\partial \boldsymbol{\Sigma}_k} \exp\left( -\frac{1}{2}(\boldsymbol{x}_n - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\boldsymbol{x}_n - \boldsymbol{\mu}_k) \right) \right] .
\tag{11.32c}
$$
`;

test('full Markdown reproduces the three old errors with source locations', async () => {
  const render = await createRenderer();
  const result = await render('regression.md', broken);
  assert.equal(result.renderedFormulas, 0);
  assert.deepEqual(result.errors.map(({ tag, line }) => ({ tag, line })), [
    { tag: '7.62', line: 5 },
    { tag: '11.32b', line: 11 },
    { tag: '11.32c', line: 16 },
  ]);
  assert.ok(result.errors.every(error => error.message.startsWith('ParseError:')));
});

test('minimal repairs render all three equations and retain both continuation tags', async () => {
  const render = await createRenderer();
  const repaired = broken.replace('</aligned}', String.raw`\end{aligned}`)
    .replace(String.raw`\tag{11.32b}`, String.raw`\right. \tag{11.32b}`)
    .replace(String.raw`\qquad + \det`, String.raw`\left. \qquad + \det`);
  const result = await render('repaired.md', repaired);
  assert.equal(result.renderedFormulas, 3);
  assert.deepEqual(result.errors, []);
});

test('Markdown code examples are not parsed as formulas; inline and fenced math are', async () => {
  const render = await createRenderer();
  const result = await render('syntax.md', [
    '# Syntax', '', '`$\\unknown$`', '', '```text', '$$\\unknown$$', '```', '',
    'Valid $x^2$ and invalid $\\unknown$ inline.', '',
    '```math', '\\unknown', '```',
  ].join('\n'));
  assert.equal(result.renderedFormulas, 1);
  assert.equal(result.errors.length, 2);
  assert.equal(result.errors[0].line, 9);
  assert.equal(result.errors[1].line, 11);
});

test('QA ignores notebook scripts and never executes embedded code chunks', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mml-qa-scripts-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, '.crossnote'));
  const sentinel = path.join(root, 'executed.txt');
  const writeSentinel = `require('node:fs').writeFileSync(${JSON.stringify(sentinel)}, 'executed')`;
  fs.writeFileSync(path.join(root, '.crossnote', 'parser.js'),
    `({ onWillParseMarkdown: async function(markdown) { ${writeSentinel}; return markdown; } })`);
  const render = await createRenderer(root);
  const result = await render('safe.md', [
    '```javascript {cmd="node" run_on_save=true}', writeSentinel, '```', '', '$$x$$',
  ].join('\n'));
  assert.equal(fs.existsSync(sentinel), false);
  assert.equal(result.renderedFormulas, 1);
  assert.deepEqual(result.errors, []);
  assert.deepEqual(fs.readdirSync(path.join(root, '.crossnote')), ['parser.js']);
});

test('CLI fails on broken math and includes file, line, and tag', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mml-qa-cli-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'broken.md');
  fs.writeFileSync(file, broken);
  const result = spawnSync(process.execPath, [path.join(__dirname, 'verify_math.cjs'), file],
    { encoding: 'utf8' });
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /broken\.md:5: \(7\.62\)/);
  assert.match(result.stderr, /broken\.md:11: \(11\.32b\)/);
  assert.match(result.stderr, /broken\.md:16: \(11\.32c\)/);
  assert.match(result.stdout, /1 files, 0 formulas, 3 errors/);
});
