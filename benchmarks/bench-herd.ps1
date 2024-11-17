cargo build --release

hyperfine --min-runs 2 --sort command `
  --setup '.\target\release\herd.exe .\benchmarks\helloworld.herd' `
  --export-json './benchmarks/bench-herd.json' `
  '.\target\release\herd.exe .\benchmarks\helloworld.herd' `
  '.\target\release\herd.exe .\benchmarks\mandelbrot.herd' `
  '.\target\release\herd.exe .\benchmarks\binarytrees.herd'

