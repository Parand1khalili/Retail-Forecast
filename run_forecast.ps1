$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$pythonExe = if (Test-Path -LiteralPath $bundledPython) { $bundledPython } else { (Get-Command python -ErrorAction Stop).Source }
$ready = $false
try {
    $reply = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
    $ready = ($reply.service -eq 'ra-forecast' -and $reply.ready)
} catch { }
if (-not $ready) {
    $scriptPath = Join-Path $projectRoot 'forecasting\app.py'
    Start-Process -FilePath $pythonExe -ArgumentList ('"' + $scriptPath + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $reply = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
            if ($reply.service -eq 'ra-forecast' -and $reply.ready) { $ready=$true; break }
        } catch { }
    }
}
if (-not $ready) { throw 'The forecast app did not start on port 8765.' }
try { Start-Process 'http://127.0.0.1:8765' }
catch { Write-Output 'Open this address in your browser: http://127.0.0.1:8765' }
