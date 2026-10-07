// node js/build.mjs: writes index.mjs (ES module) from index.js (CommonJS) so both entry points share one source.
import { readFileSync, writeFileSync } from "node:fs";
const here = new URL(".", import.meta.url);
const src = readFileSync(new URL("index.js", here), "utf8");
const marker = "module.exports = ";
const i = src.lastIndexOf(marker);
if (i < 0) throw new Error("index.js must end with module.exports = {...}");
const names = src.slice(i + marker.length).trim().replace(/;$/, "");
const body = src.slice(0, i).replace('"use strict";\n', "");
writeFileSync(new URL("index.mjs", here), `${body}export ${names};\nexport default ${names};\n`);
