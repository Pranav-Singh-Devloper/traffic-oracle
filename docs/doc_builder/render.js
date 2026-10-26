// Render content.json (from build_content.py) into a .docx. Usage: node render.js content.json out.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, ImageRun, PageBreak, LevelFormat, TableOfContents, Header, Footer, PageNumber, BorderStyle,
} = require("docx");

const [, , inPath, outPath] = process.argv;
const blocks = JSON.parse(fs.readFileSync(inPath, "utf8"));
const FONT = "Calibri";
const CONTENT_W = 9026; // A4 width 11906 − 2 × 1440 margins (DXA)
const ACCENT = "1F4E79";

// **bold**, *italic*, `code` → runs
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", color: "333333", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const caption = (t) => new Paragraph({ children: runs(t, { size: 18, italics: true, color: "444444" }), spacing: { before: 60, after: 200 } });
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(b) {
  const ncol = b.header.length;
  const rel = b.widths && b.widths.length === ncol ? b.widths : Array(ncol).fill(1);
  const sum = rel.reduce((a, c) => a + c, 0);
  const widths = rel.map((w) => Math.floor((w / sum) * CONTENT_W));
  widths[ncol - 1] += CONTENT_W - widths.reduce((a, c) => a + c, 0);
  const size = b.small ? 16 : 18;
  const mkRow = (cells, header) => new TableRow({
    tableHeader: header,
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: header ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(String(c), { size, bold: header || undefined }) })],
    })),
  });
  return [
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [mkRow(b.header, true), ...b.rows.map((r) => mkRow(r, false))] }),
    caption(b.caption),
  ];
}

function image(b) {
  const wPx = Math.round(b.width_in * 96);
  const hPx = Math.round(wPx * (b.px[1] / b.px[0]));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path), transformation: { width: wPx, height: hPx } })] }),
    caption(b.caption),
  ];
}

const children = [];
for (const b of blocks) {
  switch (b.t) {
    case "title":
      children.push(new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: b.title, bold: true, size: 64, color: ACCENT })] }));
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: b.subtitle, size: 28, color: "404040" })] }));
      for (const l of b.lines) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs(l, { size: 22 }) }));
      children.push(new Paragraph({ children: [new PageBreak()] }));
      break;
    case "toc":
      children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
      children.push(new Paragraph({ children: runs("(In Word: right-click the table of contents → Update Field to fill page numbers.)", { size: 16, italics: true, color: "777777" }) }));
      break;
    case "h1": children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.text)] })); break;
    case "h2": children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.text)] })); break;
    case "h3": children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(b.text)] })); break;
    case "p": children.push(new Paragraph({ children: runs(b.text), spacing: { after: 120 } })); break;
    case "bullets":
      for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(it), spacing: { after: 60 } }));
      break;
    case "numbered":
      for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(it), spacing: { after: 60 } }));
      break;
    case "eq":
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
        shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
        children: [new TextRun({ text: b.text, font: "Cambria Math", size: 20 })] }));
      break;
    case "callout":
      children.push(new Paragraph({ spacing: { before: 120, after: 160 },
        shading: { fill: "FFF4E5", type: ShadingType.CLEAR, color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: "E69138", space: 6 } },
        children: runs(b.text, { size: 20 }) }));
      break;
    case "table": children.push(...table(b)); break;
    case "img": children.push(...image(b)); break;
    case "pagebreak": children.push(new Paragraph({ children: [new PageBreak()] })); break;
  }
}

const doc = new Document({
  creator: "Team T042", title: "Traffic Oracle — complete technical documentation",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACCENT, font: FONT }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5C8A", font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "404040", font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Traffic Oracle · Team T042", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "888888" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log("wrote", outPath, (buf.length / 1e6).toFixed(1), "MB"); });
