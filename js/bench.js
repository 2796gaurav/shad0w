// node js/bench.js <table.s0> <texts.txt> <python_preds.txt>   -> parity + per-decision latency
"use strict";
const fs = require("fs");
const { Table } = require("./index.js");
const [tablePath, textsPath, predsPath] = process.argv.slice(2);
const buf = fs.readFileSync(tablePath);
const t = new Table(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const texts = fs.readFileSync(textsPath, "utf8").split("\n").filter((x) => x.length);
const ref = predsPath ? fs.readFileSync(predsPath, "utf8").split("\n").filter((x) => x.length).map(Number) : null;
let agree = 0;
for (let i = 0; i < texts.length; i++) if (ref && t.decide(texts[i]).index === ref[i]) agree++;
for (let r = 0; r < 3; r++) for (const x of texts) t.decide(x);  // warm up
const lat = [];
for (const x of texts) { const a = process.hrtime.bigint(); t.decide(x); lat.push(Number(process.hrtime.bigint() - a)); }
lat.sort((a, b) => a - b);
console.log(JSON.stringify({ n: texts.length, parity: ref ? agree / texts.length : null, F: t.F, K: t.K,
  p50_us: lat[lat.length >> 1] / 1000, p99_us: lat[Math.floor(lat.length * 0.99)] / 1000, bytes: buf.byteLength }));
