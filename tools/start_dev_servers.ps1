param(
    [string]$DjangoHost = "127.0.0.1",
    [int]$DjangoPort = 8000,

    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8010
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ManagePy = Join-Path $ProjectRoot "src\kaliok\ui\manage.py"

$TmpDir = Join-Path $ProjectRoot ".tmp"
$PidFile = Join-Path $TmpDir "dev_servers.json"

$DjangoUrl = "http://${DjangoHost}:${DjangoPort}"
$ApiUrl = "http://${ApiHost}:${ApiPort}"
$ApiHealthUrl = "${ApiUrl}/health"


function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}


function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port
    )

    try {
        $client = New-Object System.Net.Sockets.TcpClient

        $result = $client.BeginConnect(
            $HostName,
            $Port,
            $null,
            $null
        )

        $connected = $result.AsyncWaitHandle.WaitOne(500)

        if (-not $connected) {
            $client.Close()
            return $false
        }

        $client.EndConnect($result)
        $client.Close()

        return $true
    }
    catch {
        return $false
    }
}


function Test-FastApi {
    try {
        $response = Invoke-RestMethod `
            -Uri $ApiHealthUrl `
            -TimeoutSec 2

        return (
            $response.status -eq "ok" -and
            $response.service -eq "kaliok-api"
        )
    }
    catch {
        return $false
    }
}


function Test-Django {
    try {
        $response = Invoke-WebRequest `
            -Uri $DjangoUrl `
            -TimeoutSec 2 `
            -UseBasicParsing

        return (
            $response.StatusCode -ge 200 -and
            $response.StatusCode -lt 500
        )
    }
    catch {
        return $false
    }
}


function Wait-ForService {
    param(
        [string]$Name,
        [scriptblock]$Test,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if (& $Test) {
            Write-Host "$Name : OK" -ForegroundColor Green
            return $true
        }

        Start-Sleep -Milliseconds 500
    }

    Write-Host (
        "$Name : ECHEC apres ${TimeoutSeconds}s"
    ) -ForegroundColor Red

    return $false
}


function Test-ProcessAlive {
    param([int]$ProcessId)

    try {
        Get-Process -Id $ProcessId -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}


if (-not (Test-Path $Python)) {
    Write-Host (
        "Python virtuel introuvable : $Python"
    ) -ForegroundColor Red

    exit 1
}


if (-not (Test-Path $ManagePy)) {
    Write-Host (
        "manage.py introuvable : $ManagePy"
    ) -ForegroundColor Red

    exit 1
}


if (-not (Test-Path $TmpDir)) {
    New-Item `
        -ItemType Directory `
        -Path $TmpDir | Out-Null
}


$State = [ordered]@{
    django = $null
    api = $null
}


if (Test-Path $PidFile) {
    try {
        $ExistingState = Get-Content $PidFile -Raw |
            ConvertFrom-Json

        if (
            $ExistingState.api -and
            (Test-ProcessAlive -ProcessId $ExistingState.api.pid)
        ) {
            $State.api = $ExistingState.api
        }

        if (
            $ExistingState.django -and
            (Test-ProcessAlive -ProcessId $ExistingState.django.pid)
        ) {
            $State.django = $ExistingState.django
        }
    }
    catch {
        Write-Host (
            "Ancien fichier d'etat invalide, il sera remplace."
        ) -ForegroundColor Yellow
    }
}


Write-Host ""
Write-Host "kaliok V2 - serveurs de developpement"
Write-Host "Projet   : $ProjectRoot"
Write-Host "Django   : $DjangoUrl"
Write-Host "FastAPI  : $ApiUrl"


Write-Step "Verification de FastAPI"

if (Test-FastApi) {
    Write-Host (
        "FastAPI fonctionne deja."
    ) -ForegroundColor Green
}
else {
    if (Test-TcpPort -HostName $ApiHost -Port $ApiPort) {
        Write-Host (
            "Le port $ApiPort est occupe, " +
            "mais kaliok FastAPI ne repond pas correctement."
        ) -ForegroundColor Red

        exit 1
    }

    Write-Host "Demarrage de FastAPI..."

    $ApiArgs = @(
        "-m",
        "uvicorn",
        "kaliok.api.main:app",
        "--host",
        $ApiHost,
        "--port",
        "$ApiPort"
    )

    $ApiProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList $ApiArgs `
        -WorkingDirectory $ProjectRoot `
        -PassThru

    $State.api = [ordered]@{
        pid = $ApiProcess.Id
        host = $ApiHost
        port = $ApiPort
        url = $ApiUrl
    }

    if (-not (
        Wait-ForService `
            -Name "FastAPI" `
            -Test ${function:Test-FastApi}
    )) {
        if (Test-ProcessAlive -ProcessId $ApiProcess.Id) {
            Stop-Process -Id $ApiProcess.Id -Force
        }

        exit 1
    }
}


Write-Step "Verification de Django"

if (Test-Django) {
    Write-Host (
        "Django fonctionne deja."
    ) -ForegroundColor Green
}
else {
    if (Test-TcpPort -HostName $DjangoHost -Port $DjangoPort) {
        Write-Host (
            "Le port $DjangoPort est occupe, " +
            "mais Django ne repond pas correctement."
        ) -ForegroundColor Red

        exit 1
    }

    Write-Host "Demarrage de Django..."

    $OldApiBaseUrl = $env:KALIOK_API_BASE_URL
    $env:KALIOK_API_BASE_URL = $ApiUrl

    try {
        $DjangoArgs = @(
            $ManagePy,
            "runserver",
            "${DjangoHost}:${DjangoPort}",
            "--noreload"
        )

        $DjangoProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $DjangoArgs `
            -WorkingDirectory $ProjectRoot `
            -PassThru
    }
    finally {
        if ($null -eq $OldApiBaseUrl) {
            Remove-Item Env:KALIOK_API_BASE_URL `
                -ErrorAction SilentlyContinue
        }
        else {
            $env:KALIOK_API_BASE_URL = $OldApiBaseUrl
        }
    }

    $State.django = [ordered]@{
        pid = $DjangoProcess.Id
        host = $DjangoHost
        port = $DjangoPort
        url = $DjangoUrl
    }

    if (-not (
        Wait-ForService `
            -Name "Django" `
            -Test ${function:Test-Django}
    )) {
        if (Test-ProcessAlive -ProcessId $DjangoProcess.Id) {
            Stop-Process -Id $DjangoProcess.Id -Force
        }

        exit 1
    }
}


$State |
    ConvertTo-Json -Depth 5 |
    Set-Content `
        -Path $PidFile `
        -Encoding UTF8


Write-Step "Verification finale"

$FastApiOk = Test-FastApi
$DjangoOk = Test-Django

if ($FastApiOk -and $DjangoOk) {
    Write-Host ""
    Write-Host (
        "Les deux serveurs kaliok sont operationnels."
    ) -ForegroundColor Green

    Write-Host ""
    Write-Host "Interface Django : $DjangoUrl"
    Write-Host "API FastAPI      : $ApiUrl"
    Write-Host "API health       : $ApiHealthUrl"
    Write-Host "Documentation API: ${ApiUrl}/docs"
    Write-Host ""
    Write-Host "Arret :"
    Write-Host ".\tools\stop_dev_servers.ps1"

    exit 0
}


Write-Host ""
Write-Host (
    "Au moins un service kaliok n'est pas operationnel."
) -ForegroundColor Red

exit 1