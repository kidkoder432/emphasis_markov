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

# Run each script and capture output
Write-Host "Running eval_ablative.py..."
py eval_ablative.py $num $name | Tee-Object -FilePath $logFile -Append

# Subsequent runs append to the same file
Write-Host "Running eval_ablative_2.py..."
py eval_ablative_2.py $num $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running eval_ablative_3.py..."
py eval_ablative_3.py $num $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running eval_phrases_2.py..."
py eval_phrases_2.py $num $name | Tee-Object -FilePath $logFile -Append