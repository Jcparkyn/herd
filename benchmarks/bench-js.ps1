
hyperfine --min-runs 2 --sort command `
  --export-json './benchmarks/bench-js.json' `
  'node .\benchmarks\binarytrees.js' `
  'node .\benchmarks\helloworld.js' `
  'node .\benchmarks\mandelbrot.js'

./benchmarks/compare.ps1