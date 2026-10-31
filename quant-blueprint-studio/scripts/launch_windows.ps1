$ErrorActionPreference = 'Stop'
$engine = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList 'backend/server.py' -PassThru -NoNewWindow
try {
  & .venv\Scripts\python.exe scripts/wait_backend.py $engine.Id
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & npm.cmd run electron:dev
  $result = $LASTEXITCODE
} finally {
  if (-not $engine.HasExited) { Stop-Process -Id $engine.Id }
}
exit $result
