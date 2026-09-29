// Writes src/mocks/fixtures/synthetic-house.pdf: a tiny, entirely synthetic
// three-page PDF for the UI mocks and Playwright (rule K2: no house data).
// Page 2 has a letter-spaced heading ("D E S C R I P T I O N") drawn one glyph
// at a time, so the viewer's fuzzy highlight is exercised the way the real
// fixture corpus will exercise it. Run with `npm run fixtures:pdf`.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const pages = [
  {
    lines: [
      [72, 720, 18, "SYNTHETIC HOME INSPECTION REPORT"],
      [72, 690, 11, "Property: 12 Example Lane, Testville"],
      [72, 674, 11, "Inspection date: June 22, 2023"],
      [72, 658, 11, "Inspector: Sample Inspections Co."],
      [72, 620, 11, "Summary: The roof covering is composition shingle in fair condition."],
      [72, 604, 11, "The water heater is a 50 gallon gas unit with a seismic strap."],
      [72, 60, 9, "Page 1 of 3"],
    ],
  },
  {
    spaced: [72, 720, 16, "DESCRIPTION", 17],
    lines: [
      [72, 690, 11, "The furnace is a forced-air gas unit located in the garage."],
      [72, 674, 11, "It was installed in 2009 and is near the end of its expected life."],
      [72, 658, 11, "A flexible gas connector is routed through the cabinet wall."],
      [72, 642, 11, "Recommend evaluation by a licensed HVAC contractor."],
      [72, 60, 9, "Page 2 of 3"],
    ],
  },
  {
    lines: [
      [72, 720, 16, "INVOICE 1001 - Brightline Pest Co."],
      [72, 690, 11, "Termite damage repair at the rear deck ledger."],
      [72, 674, 11, "Total due: $1,234.00"],
      [72, 658, 11, "Completed July 14, 2023."],
      [72, 60, 9, "Page 3 of 3"],
    ],
  },
];

const esc = (s) => s.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");

function content(page) {
  const ops = [];
  if (page.spaced) {
    const [x, y, size, word, step] = page.spaced;
    [...word].forEach((ch, i) => {
      ops.push(`BT /F1 ${size} Tf 1 0 0 1 ${x + i * step} ${y} Tm (${esc(ch)}) Tj ET`);
    });
  }
  for (const [x, y, size, text] of page.lines) {
    ops.push(`BT /F1 ${size} Tf 1 0 0 1 ${x} ${y} Tm (${esc(text)}) Tj ET`);
  }
  return ops.join("\n");
}

const objects = [];
const add = (body) => {
  objects.push(body);
  return objects.length;
};
const catalog = add(null);
const pagesObj = add(null);
const font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
const kids = [];
for (const page of pages) {
  const stream = content(page);
  const contentObj = add(`<< /Length ${Buffer.byteLength(stream, "latin1")} >>\nstream\n${stream}\nendstream`);
  kids.push(
    add(
      `<< /Type /Page /Parent ${pagesObj} 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 ${font} 0 R >> >> /Contents ${contentObj} 0 R >>`,
    ),
  );
}
objects[catalog - 1] = `<< /Type /Catalog /Pages ${pagesObj} 0 R >>`;
objects[pagesObj - 1] = `<< /Type /Pages /Kids [${kids.map((k) => `${k} 0 R`).join(" ")}] /Count ${kids.length} >>`;

let out = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";
const offsets = [];
objects.forEach((body, i) => {
  offsets.push(Buffer.byteLength(out, "latin1"));
  out += `${i + 1} 0 obj\n${body}\nendobj\n`;
});
const xref = Buffer.byteLength(out, "latin1");
out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
for (const off of offsets) out += `${String(off).padStart(10, "0")} 00000 n \n`;
out += `trailer\n<< /Size ${objects.length + 1} /Root ${catalog} 0 R /Info << /Title (Synthetic house fixture) /Producer (hal9000 web fixtures) >> >>\nstartxref\n${xref}\n%%EOF\n`;

const target = fileURLToPath(new URL("../src/mocks/fixtures/synthetic-house.pdf", import.meta.url));
writeFileSync(target, Buffer.from(out, "latin1"));
console.log(`wrote ${target} (${Buffer.byteLength(out, "latin1")} bytes)`);
