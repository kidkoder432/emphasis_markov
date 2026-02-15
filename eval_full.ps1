# Set the number
$name = if ($args.Count -ge 1) { $args[0] } else { "amrit" }
# Get the raga name from command line arguments if provided
$num = if ($args.Count -ge 2) { $args[1] } else { 100 }

# Log file
$logFile = "eval_log_$name.txt"

# Clear the log file before starting
Write-Host "Clearing existing log file..."
Clear-Content -Path $logFile

Write-Host "Evaluating raga: $name"

Write-Host "Running evals..."
py eval_full.py $name $num $args | Tee-Object -FilePath $logFile -Append