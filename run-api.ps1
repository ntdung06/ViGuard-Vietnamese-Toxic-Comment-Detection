$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:CHEAPKEYAI_API_KEY) {
    $apiSecret = Read-Host 'Nhap CheapKeyAI API key' -AsSecureString
    $env:CHEAPKEYAI_API_KEY = [System.Net.NetworkCredential]::new('', $apiSecret).Password
}
$env:PREDICTION_BACKEND = 'remote'
if (-not $env:CHEAPKEYAI_BASE_URL) { $env:CHEAPKEYAI_BASE_URL = 'https://cheapkeyai.shop/v1' }
if (-not $env:CHEAPKEYAI_MODEL) { $env:CHEAPKEYAI_MODEL = 'gpt-4o-mini' }
$projectPython = if (Test-Path -LiteralPath '.venv/Scripts/python.exe') {
    Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
} else { 'python' }
& $projectPython manage.py migrate --noinput
if ($LASTEXITCODE -ne 0) { throw 'Migration failed' }
& $projectPython manage.py runserver 127.0.0.1:8000 --noreload
