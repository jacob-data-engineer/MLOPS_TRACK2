# Run this from an ELEVATED (Administrator) PowerShell window.
# Compacts Docker Desktop's WSL2 virtual disk after `docker builder prune`/
# `docker image prune` freed space inside it -- that freed space doesn't
# shrink the actual .vhdx file on the Windows host until compacted manually.

$vhdxPath = "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx"

Write-Host "Trimming freed blocks inside the docker-desktop WSL filesystem (deleting files inside..."
Write-Host "WSL's ext4 filesystem doesn't tell the virtual disk they're free until this runs)..."
wsl -d docker-desktop -- fstrim -av

Write-Host "Shutting down WSL (stops Docker Desktop too; it restarts fine afterward)..."
wsl --shutdown
Start-Sleep -Seconds 5

Write-Host "Size before: $([math]::Round((Get-Item $vhdxPath).Length / 1GB, 2)) GB"

$diskpartScript = @"
select vdisk file="$vhdxPath"
attach vdisk readonly
compact vdisk
detach vdisk
"@
$scriptPath = "$env:TEMP\compact_vhdx.txt"
Set-Content -Path $scriptPath -Value $diskpartScript -Encoding ASCII
diskpart /s $scriptPath

Write-Host "Size after: $([math]::Round((Get-Item $vhdxPath).Length / 1GB, 2)) GB"
Write-Host "Restarting Docker Desktop..."
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
