param([switch]$CheckOnly, [ValidateRange(1024, 65535)][int]$Port = 8766)
# This forecast service deliberately uses a different port from the Juice
# Intelligence dashboard (8765). A separate port can be passed with -Port.
$forecastUrl = "http://127.0.0.1:$Port"
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# Prefer a usable local environment, but never silently launch an environment
# where CatBoost cannot be imported. The launcher picks the SAME interpreter
# that passed the dependency import check.
$pythonCandidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $projectRoot 'venv\Scripts\python.exe')
)
$activePython = Get-Command python -ErrorAction SilentlyContinue
if ($activePython) { $pythonCandidates += $activePython.Source }
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $pythonFromLauncher = & $pyLauncher.Source -3 -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $pythonFromLauncher) {
        $pythonCandidates += (($pythonFromLauncher | Select-Object -Last 1).Trim())
    }
}
$preflightPath = Join-Path $projectRoot 'forecasting\runtime_preflight.py'
$pythonExe = $null
$firstProblem = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $raw = & $candidate -B $preflightPath 2>$null
    if (-not $raw) { continue }
    try { $diagnosis = ($raw | Out-String | ConvertFrom-Json) }
    catch { continue }
    if ($diagnosis.ready) {
        $pythonExe = $candidate
        break
    }
    if (-not $firstProblem) { $firstProblem = $diagnosis }
}
if (-not $pythonExe) {
    if ($firstProblem) {
        Write-Host ('Python: ' + $firstProblem.python_executable)
        Write-Host ('Dependency error: ' + ($firstProblem.failures | ConvertTo-Json -Compress))
        Write-Host ('Install dependencies using this EXACT interpreter:')
        Write-Host $firstProblem.install_command
    }
    throw 'No Python installation with working NumPy, Pandas and CatBoost found. Install requirements into the chosen Python and retry.'
}
Write-Host ('EMRC Forecast Python: ' + $pythonExe)
if ($CheckOnly) { Write-Host 'EMRC_FORECAST_PREFLIGHT_OK'; exit 0 }
$expected = (& $pythonExe -B (Join-Path $projectRoot 'forecasting\runtime_identity.py')) | ConvertFrom-Json
function Read-Health {
    try { Invoke-RestMethod ($forecastUrl + '/api/health') -TimeoutSec 2 } catch { $null }
}
function Is-Current($reply) {
    $reply -and $reply.service -eq 'ra-forecast' -and $reply.ready -and
    $reply.runtime.root -eq $expected.root -and $reply.runtime.source_id -eq $expected.source_id -and
    $reply.runtime.python -eq $expected.python -and $reply.runtime.port -eq $Port
}
$reply = Read-Health
# A foreign service may not expose /api/health at all. Detect the listener
# before trying to launch another Python process and never kill it.
if (-not $reply) {
    $existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($existingListener) {
        throw "Port $Port is already occupied by another process. Do not stop an unknown server. Use run_forecast.ps1 -Port 8876 or inspect Get-NetTCPConnection -LocalPort $Port."
    }
}
if ($reply -and -not (Is-Current $reply)) {
    if ($reply.runtime.root -ne $expected.root) {
        throw "Port $Port belongs to an unverified/other server. Do not stop it. Choose another port with -Port 8876 or inspect the process owner."
    }
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen
    foreach ($ownerId in ($listeners.OwningProcess | Select-Object -Unique)) {
        $owner = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $ownerId)
        $appPath = Join-Path $projectRoot 'forecasting\app.py'
        if ($ownerId -ne $reply.runtime.pid -or -not $owner.CommandLine.Contains($appPath)) {
            throw 'Server owner does not match the verified app path; stop it manually after checking.'
        }
        Stop-Process -Id $ownerId
    }
    $reply = $null
}
if (-not (Is-Current $reply)) {
    $scriptPath = Join-Path $projectRoot 'forecasting\app.py'
    Start-Process -FilePath $pythonExe -ArgumentList ('-B "' + $scriptPath + '" --port ' + $Port) -WorkingDirectory $projectRoot -WindowStyle Hidden
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        $reply = Read-Health
        if (Is-Current $reply) { break }
    }
}
if (-not (Is-Current $reply)) { throw "The current EMRC Forecast app did not start on port $Port. Check the port owner and Python environment." }
Write-Host ('EMRC Retail Forecast URL: ' + $forecastUrl)
try { Start-Process $forecastUrl }
catch { Write-Output ('Open this address in your browser: ' + $forecastUrl) }
