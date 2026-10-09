# AIRE: primer paso de la instalación (lo lanza la extensión de SketchUp, oculto).
# Descarga uv (un único ejecutable que trae su propio Python), prepara el Python
# del backend y pasa el testigo al instalador en Python, que hace el resto.
param(
  [Parameter(Mandatory = $true)][string]$AireHome,
  [Parameter(Mandatory = $true)][string]$Backend,
  [Parameter(Mandatory = $true)][string]$Progress
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest es mucho más rápido sin barra
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Steps = @(
  'Preparando el instalador',
  'Comprobando tu ordenador',
  'Descargando el motor de render',
  'Instalando el motor de render',
  'Descargando el modelo de inteligencia artificial',
  'Probando que todo funciona'
)

function Write-State([string]$State, [string]$Detail, $ErrorText) {
  $obj = [ordered]@{
    state = $State; steps = $Steps; step = 0; title = $Steps[0]; detail = $Detail; percent = $null
    pid = $PID; updated = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); error = $ErrorText; result = $null
  }
  $json = $obj | ConvertTo-Json -Compress
  $tmp = "$Progress.tmp"
  [IO.File]::WriteAllText($tmp, $json, (New-Object Text.UTF8Encoding $false))
  Move-Item -Force -LiteralPath $tmp -Destination $Progress
}

# Ejecuta un programa sin ventana y añade su salida al registro. Devuelve el código de salida.
# (Con Start-Process se evita que PowerShell 5.1 trate como error lo que uv escribe en stderr.)
function Invoke-Logged([string]$Exe, [string[]]$ArgList) {
  $quoted = $ArgList | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
  $out = Join-Path $LogDir 'last-out.txt'
  $err = Join-Path $LogDir 'last-err.txt'
  Add-Content -LiteralPath $Log -Value "`n$ $Exe $($quoted -join ' ')"
  $p = Start-Process -FilePath $Exe -ArgumentList ($quoted -join ' ') -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput $out -RedirectStandardError $err
  Get-Content -LiteralPath $out, $err -ErrorAction SilentlyContinue | Add-Content -LiteralPath $Log
  return $p.ExitCode
}

try {
  $LogDir = Join-Path $AireHome 'logs'
  New-Item -ItemType Directory -Force -Path $AireHome, $LogDir, (Split-Path -Parent $Progress) | Out-Null
  $Log = Join-Path $LogDir 'install.log'
  Add-Content -LiteralPath $Log -Value "`n=== $(Get-Date -Format s) bootstrap"

  # uv y su Python quedan dentro de la carpeta de AIRE (desinstalar = borrar la carpeta)
  $env:UV_PYTHON_INSTALL_DIR = Join-Path $AireHome 'python'
  $env:UV_CACHE_DIR = Join-Path $AireHome 'cache'
  $env:UV_LINK_MODE = 'copy'

  $UvDir = Join-Path $AireHome 'uv'
  $Uv = Join-Path $UvDir 'uv.exe'
  if (-not (Test-Path -LiteralPath $Uv)) {
    Write-State 'running' 'Descargando el instalador…' $null
    $zip = Join-Path $AireHome 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    Expand-Archive -Force -LiteralPath $zip -DestinationPath $UvDir
    Remove-Item -LiteralPath $zip
  }

  $EnvDir = Join-Path $AireHome 'env'
  $Py = Join-Path $EnvDir 'Scripts\python.exe'
  if (-not (Test-Path -LiteralPath $Py)) {
    Write-State 'running' 'Preparando Python…' $null
    if ((Invoke-Logged $Uv @('venv', $EnvDir, '--python', '3.12')) -ne 0) { throw 'uv venv' }
  }

  Write-State 'running' 'Instalando componentes…' $null
  if ((Invoke-Logged $Uv @('pip', 'install', '--python', $Py, 'numpy', 'numba', 'pillow')) -ne 0) { throw 'uv pip install' }

  # El instalador en Python escribe su propio progreso a partir de aquí
  $code = Invoke-Logged $Py @((Join-Path $Backend 'run.py'), 'installer', '--home', $AireHome, '--uv', $Uv, '--progress', $Progress)
  exit $code
}
catch {
  Add-Content -LiteralPath $Log -Value ($_ | Out-String) -ErrorAction SilentlyContinue
  Write-State 'error' '' ('No se ha podido preparar AIRE. Comprueba la conexión a internet y vuelve a intentarlo. (' + $_.Exception.Message + ')')
  exit 1
}
