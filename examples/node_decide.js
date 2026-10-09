// node examples/node_decide.js examples/out/quickstart/bundle "my card was stolen"
// Certified answers are served from the table; everything else would go to your model.
const { Bundle } = require("../js/index.js");

(async () => {
  const [dir, text = ""] = process.argv.slice(2);
  const bundle = await Bundle.load(dir);
  const t0 = process.hrtime.bigint();
  const { answers } = bundle.decide(text);
  const us = Number(process.hrtime.bigint() - t0) / 1000;
  for (const [name, a] of Object.entries(answers)) {
    const action = a.certified ? "serve" : "defer to your model";
    console.log(JSON.stringify({ question: name, choice: a.choice ?? a.answer, confidence: +a.confidence.toFixed(4), certified: a.certified, action, us }));
  }
})();
