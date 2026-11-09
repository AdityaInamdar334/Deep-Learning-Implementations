// Exercise the pure recall verifier from a user-supplied v2.0.33 checkout.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const checkout = process.argv[2];
if (!checkout) throw new Error("Usage: node recall.mjs /path/to/evolver");
const file = resolve(checkout, "packages/evolver-core/dist/ops/recall.js");
const { verifyGeneRecall } = await import(pathToFileURL(file));
const strategy = "Inspect partition metadata before counting filtered tokens.";
const gene = { geneId: "tutorial-partition-check", strategy: [strategy] };
const cases = [
  ["Only injected", [{role:"user",text:strategy}], "unknown"],
  ["Different reply", [{role:"user",text:strategy},{role:"assistant",text:"I will examine the configuration."}], "unused"],
  ["Promised action", [{role:"assistant",text:"I plan to inspect partition metadata before counting filtered tokens."}], "used"],
  ["Negated action", [{role:"assistant",text:"I did not inspect partition metadata before counting filtered tokens."}], "used"],
];
for (const [name, turns, expected] of cases) {
  const result = verifyGeneRecall(gene, turns);
  console.log(name, result.recalled, result.score.toFixed(2));
  assert.equal(result.recalled, expected);
}
