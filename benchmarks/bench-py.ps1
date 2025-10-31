
hyperfine --min-runs 2 --sort command `
  --export-json './benchmarks/bench-py.json' `
  'py -3.11 .\benchmarks\nbody.py' `
  'py -3.11 .\benchmarks\binarytrees.py' `
  'py -3.11 .\benchmarks\helloworld.py' `
  'py -3.11 .\benchmarks\iterables.py' `
  'py -3.11 .\benchmarks\csv.py' `
  'py -3.11 .\benchmarks\mandelbrot.py'

./benchmarks/compare.ps1
