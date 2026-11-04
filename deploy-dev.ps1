[CmdletBinding()]
param(
    [string]$Profile,
    [string]$TerraformDirectory
)

$ErrorActionPreference = 'Stop'
$bashPath = Join-Path $env:ProgramFiles 'Git\bin\bash.exe'
if (-not (Test-Path -LiteralPath $bashPath)) {
    throw 'Install Git for Windows (including Git Bash), then retry.'
}
if ($TerraformDirectory) {
    $resolvedTools = (Resolve-Path -LiteralPath $TerraformDirectory).Path
    $env:PATH = "$resolvedTools;$env:PATH"
}
foreach ($tool in @('aws', 'terraform')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing $tool on PATH. Install it or add its directory to PATH before retrying."
    }
}
if ($Profile) { $env:AWS_PROFILE = $Profile }
& aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) {
    throw 'AWS authentication failed. Sign in using your organization-approved AWS profile, then retry with -Profile if needed.'
}

# The existing deployment builds in CodeBuild; local Node and Docker are unnecessary.
$env:PORT = '8090'
$env:CONTAINER = 'track2-app-dev'
$env:APP_ROLE = 'dev'
$env:TRACK2_RECORD_BACKEND = 'postgres'
$env:CATALOG_SCHEMA = 'track2_dev'
$env:SKIP_LAMBDA = '1'
Push-Location $PSScriptRoot
try {
    & $bashPath ./deploy.sh
    if ($LASTEXITCODE -ne 0) { throw "Development deployment failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
