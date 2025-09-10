
# Get the raga name from command line arguments if provided
$name = if ($args.Count -ge 1) { $args[0] } else { "amrit" }

# Log file
$logFile = "training_log_$name.txt"

# Clear the log file before starting
Write-Host "Clearing existing log file..."
Clear-Content -Path $logFile

Write-Host "Running training scripts for raga: $name"

# Run each script and capture output
Write-Host "Running loadjson.py..."
py ./training/loadjson.py $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running maketpm.py..."
py ./training/maketpm.py $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running learn_emphasis.py..."
py ./training/learn_emphasis.py $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running tags.py..."
py ./training/tags.py $name | Tee-Object -FilePath $logFile -Append

Write-Host "Running tags_tpm_gen.py..."
py ./training/tags_tpm_gen.py $name | Tee-Object -FilePath $logFile -Append