# Set the number
$num = if ($args.Count -ge 1) { $args[0] } else { 100 }
# Get the raga name from command line arguments if provided
$name = if ($args.Count -ge 2) { $args[1] } else { "amrit" }

# Log file
$logFile = "eval_log_$name.txt"

# Clear the log file before starting
Write-Host "Clearing existing log file..."
Clear-Content -Path $logFile

Write-Host "Evaluating raga: $name"

Write-Host "Running evals..."
py eval_full.py $num $name | Tee-Object -FilePath $logFile -Append