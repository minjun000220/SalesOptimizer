// 문서 빌드: src.html 의 \( … \) (본문 수식), \[ … \] (단독 수식) 을 KaTeX 로 미리 그려서 PDF 로 출력
const fs = require('fs');
const path = require('path');
const katex = require('katex');
const { chromium } = require('playwright');

const DIR = __dirname;
const src = fs.readFileSync(path.join(DIR, 'src.html'), 'utf-8');
const macros = { '\\E': '\\mathbb{E}', '\\P': '\\mathbb{P}', '\\R': '\\mathbb{R}', '\\1': '\\mathbf{1}' };
let errors = 0;
const render = (tex, display) => {
  try {
    return katex.renderToString(tex, { displayMode: display, throwOnError: true, macros, strict: false });
  } catch (e) {
    errors++;
    console.error('KaTeX 오류:', e.message, '\n  →', tex.slice(0, 120));
    return `<span style="color:red">[수식 오류]</span>`;
  }
};
let html = src
  .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => render(t.trim(), true))
  .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => render(t.trim(), false));
const css = fs.readFileSync(path.join(DIR, 'node_modules/katex/dist/katex.min.css'), 'utf-8')
  .replace(/url\(fonts\//g, `url(file://${path.join(DIR, 'node_modules/katex/dist/fonts')}/`);
html = html.replace('/*KATEX_CSS*/', css);
fs.writeFileSync(path.join(DIR, 'out.html'), html);
if (errors) { console.error(`수식 오류 ${errors}개`); process.exit(1); }

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.join(DIR, 'out.html'), { waitUntil: 'networkidle' });
  await p.evaluate(() => document.fonts.ready);
  const out = process.argv[2] || path.join(DIR, 'manual.pdf');
  await p.pdf({
    path: out, format: 'A4', printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:7.5px;color:#888;width:100%;padding:0 16mm;font-family:Noto Sans CJK KR">SalesOptimizer · 아이돌 굿즈 안전 완판 모델 설명서</div>',
    footerTemplate: '<div style="font-size:8px;color:#888;width:100%;text-align:center;font-family:Noto Sans CJK KR"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await b.close();
  console.log('PDF:', out);
})();
