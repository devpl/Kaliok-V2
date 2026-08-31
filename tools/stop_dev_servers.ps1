$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ProjectRoot ".tmp\dev_servers.json"


function Stop-KaliokProcess {
    param(
        [string]$Name,
        [object]$State
    )

    if (-not $State) {
        Write-Host (
            "$Name : aucun processus enregistre."
        ) -ForegroundColor Yellow

        return
    }

    $ProcessId = [int]$State.pid

    try {
        $Process = Get-Process `
            -Id $ProcessId `
            -ErrorAction Stop
    }
    catch {
        Write-Host (
            "$Name : processus $ProcessId deja arrete."
        ) -ForegroundColor Yellow

        return
    }

    Write-Host (
        "Arret de $Name - PID $ProcessId..."
    )

    try {
        Stop-Process `
            -Id $ProcessId `
            -ErrorAction Stop

        try {
            Wait-Process `
                -Id $ProcessId `
                -Timeout 5 `
                -ErrorAction Stop
        }
        catch {
            if (
                Get-Process `
                    -Id $ProcessId `
                    -ErrorAction SilentlyContinue
            ) {
                Write-Host (
                    "$Name ne s'arrete pas proprement, " +
                    "arret force."
                ) -ForegroundColor Yellow

                Stop-Process `
                    -Id $ProcessId `
                    -Force `
                    -ErrorAction Stop
            }
        }

        Write-Host (
            "$Name : arrete."
        ) -ForegroundColor Green
    }
    catch {
        Write-Host (
            "$Name : impossible d'arreter le PID $ProcessId."
        ) -ForegroundColor Red

        throw
    }
}


if (-not (Test-Path $PidFile)) {
    Write-Host ""
    Write-Host (
        "Aucun serveur kaliok enregistre."
    ) -ForegroundColor Yellow

    Write-Host (
        "Fichier absent : $PidFile"
    )

    exit 0
}


try {
    $State = Get-Content `
        $PidFile `
        -Raw |
        ConvertFrom-Json
}
catch {
    Write-Host (
        "Impossible de lire le fichier d'etat : $PidFile"
    ) -ForegroundColor Red

    exit 1
}


Write-Host ""
Write-Host "kaliok V2 - arret des serveurs de developpement"
Write-Host ""

Stop-KaliokProcess `
    -Name "Django" `
    -State $State.django

Stop-KaliokProcess `
    -Name "FastAPI" `
    -State $State.api


Remove-Item `
    $PidFile `
    -Force `
    -ErrorAction SilentlyContinue


Write-Host ""
Write-Host (
    "Serveurs kaliok arretes."
) -ForegroundColor Green