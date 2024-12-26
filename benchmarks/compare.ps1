# ANSI escape codes for colors
$red = [char]27 + "[91m"
$green = [char]27 + "[92m"
$reset = [char]27 + "[0m"

# Get all benchmark files
$benchmarkFiles = Get-ChildItem -Path "./benchmarks/bench-*.json"
$results = @()

# Process baseline file first to establish reference times
$baselineFile = $benchmarkFiles | Where-Object { $_.BaseName -eq 'bench-herd-baseline' }
$baselineTimes = @{}

if ($baselineFile) {
    $jsonContent = Get-Content $baselineFile.FullName | ConvertFrom-Json
    foreach ($result in $jsonContent.results) {
        $benchmarkName = [System.IO.Path]::GetFileNameWithoutExtension(($result.command -split ' ')[-1])
        $baselineTimes[$benchmarkName] = $result.mean * 1000  # Convert to milliseconds
    }
}

# Process all benchmark files
foreach ($file in $benchmarkFiles) {
    $language = $file.BaseName -replace '^bench-',''
    $jsonContent = Get-Content $file.FullName | ConvertFrom-Json
    
    foreach ($result in $jsonContent.results) {
        $benchmarkName = [System.IO.Path]::GetFileNameWithoutExtension(($result.command -split ' ')[-1])
        $timeMs = $result.mean * 1000  # Convert to milliseconds
        
        # Find or create result object for this benchmark
        $benchmarkResult = $results | Where-Object { $_.Benchmark -eq $benchmarkName }
        if (-not $benchmarkResult) {
            $benchmarkResult = [PSCustomObject]@{ Benchmark = $benchmarkName }
            $results += $benchmarkResult
        }
        
        # Format with relative performance if this isn't baseline
        if ($language -eq 'herd-baseline') {
            $formattedValue = "{0:F1}ms" -f $timeMs
        } else {
            $baselineTime = $baselineTimes[$benchmarkName]
            $percentDiff = (($timeMs - $baselineTime) / $baselineTime) * 100
            
            # Add color based on performance
            $color = if ($percentDiff -le -1) { $red } elseif ($percentDiff -ge 1) { $green } else { $reset }
            $formattedValue = "$color{0:F1}ms ({1:F1}%)$reset" -f $timeMs, $percentDiff
        }
        
        # Add or update the property
        if ($benchmarkResult.PSObject.Properties.Name -contains $language) {
            $benchmarkResult.$language = $formattedValue
        } else {
            $benchmarkResult | Add-Member -NotePropertyName $language -NotePropertyValue $formattedValue
        }
    }
}

# Sort by benchmark name and ensure baseline column comes first
$columns = @('Benchmark')
$columns += 'herd-baseline'
$columns += ($results[0].PSObject.Properties.Name | Where-Object { $_ -ne 'Benchmark' -and $_ -ne 'herd-baseline' } | Sort-Object)

# Display results
$results | Sort-Object Benchmark | Format-Table -Property $columns -AutoSize