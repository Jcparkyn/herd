cargo build --release

hyperfine --min-runs 2 --sort command `
  --setup '.\target\release\herd.exe .\benchmarks\helloworld.herd' `
  --export-json './benchmarks/bench-herd.json' `
  '.\target\release\herd.exe .\benchmarks\nbody.herd' `
  '.\target\release\herd.exe .\benchmarks\binarytrees.herd' `
  '.\target\release\herd.exe .\benchmarks\binarytrees-m.herd' `
  '.\target\release\herd.exe .\benchmarks\helloworld.herd' `
  '.\target\release\herd.exe .\benchmarks\iterables.herd' `
  '.\target\release\herd.exe .\benchmarks\csv.herd' `
  '.\target\release\herd.exe .\benchmarks\csv-m.herd' `
  '.\target\release\herd.exe .\benchmarks\mandelbrot.herd'

./benchmarks/compare.ps1
