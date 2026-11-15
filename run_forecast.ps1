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
$expected = (& $pythonExe -B (Join-Path $projectRoot 'forecasting\runtime_identity.py')) | ConvertFrom-Json
function Read-Health {
    try { Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 2 } catch { $null }
}
function Is-Current($reply) {
    $reply -and $reply.service -eq 'ra-forecast' -and $reply.ready -and
    $reply.runtime.root -eq $expected.root -and $reply.runtime.source_id -eq $expected.source_id -and
    $reply.runtime.python -eq $expected.python
}
$reply = Read-Health
if ($reply -and -not (Is-Current $reply)) {
    if ($reply.runtime.root -ne $expected.root) {
        throw 'Port 8765 belongs to an unverified or legacy server. Verify its Python process/project path and stop it before launching this project. It will not be reused or terminated automatically.'
    }
    $listeners = Get-NetTCPConnection -LocalPort 8765 -State Listen
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
    Start-Process -FilePath $pythonExe -ArgumentList ('-B "' + $scriptPath + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        $reply = Read-Health
        if (Is-Current $reply) { break }
    }
}
if (-not (Is-Current $reply)) { throw 'The current forecast app did not start on port 8765. Check the port owner and Python environment.' }
try { Start-Process 'http://127.0.0.1:8765' }
catch { Write-Output 'Open this address in your browser: http://127.0.0.1:8765' }
