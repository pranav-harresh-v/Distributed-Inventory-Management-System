<#
.SYNOPSIS
    Generate the Python protobuf/gRPC stubs for this project.

.DESCRIPTION
    Compiles every .proto under proto/ that has content, emitting
    <name>_pb2.py and <name>_pb2_grpc.py next to the source .proto file.

    protoc is invoked with the repository root as the include path (-I.) so
    the generated gRPC module imports its message module as
    "from proto import client_pb2". That keeps the stubs importable as
    "proto.client_pb2" from anywhere in the project, rather than only when
    proto/ itself happens to be on sys.path.

    The generated files are build artifacts: they are gitignored, never
    edited by hand, and this script is the only thing that writes them.
    Deleting them and re-running this script recreates them exactly.

.PARAMETER Python
    Python interpreter to use. Defaults to the active virtual environment,
    then a venv next to or inside the repository, then "python" on PATH.

.PARAMETER Clean
    Delete existing generated stubs before regenerating.

.EXAMPLE
    .\scripts\generate_proto.ps1

.EXAMPLE
    .\scripts\generate_proto.ps1 -Clean -Python C:\path\to\python.exe
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Resolve the repository root from this script's own location, so the script
# behaves identically whether it is run from the repo root or elsewhere.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

function Resolve-PythonExe {
    param([string]$Explicit)

    if ($Explicit) {
        if (-not (Test-Path $Explicit)) {
            throw "Python interpreter not found: $Explicit"
        }
        return (Resolve-Path $Explicit).Path
    }

    # An activated virtual environment wins.
    if ($env:VIRTUAL_ENV) {
        $candidate = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    # Then a venv inside the repo, then one beside it (this project keeps its
    # virtual environment one level above the repository root).
    $candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $RepoRoot) "venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    throw "No Python interpreter found. Activate the virtual environment or pass -Python."
}

$PythonExe = Resolve-PythonExe -Explicit $Python
Write-Host "Python      : $PythonExe"
Write-Host "Repository  : $RepoRoot"

# grpc_tools ships its own protoc, so nothing needs to be installed separately.
& $PythonExe -c "import grpc_tools.protoc" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "grpc_tools is not installed for $PythonExe. Run: python -m pip install -r requirements.txt"
}

# Skip the placeholder .proto files that have no content yet; protoc rejects
# an empty file for missing "syntax", which would fail the whole run.
$ProtoDir = Join-Path $RepoRoot "proto"
$ProtoFiles = Get-ChildItem -Path $ProtoDir -Filter "*.proto" |
    Where-Object { $_.Length -gt 0 } |
    Sort-Object Name

if ($ProtoFiles.Count -eq 0) {
    throw "No non-empty .proto files found in $ProtoDir"
}

Push-Location $RepoRoot
try {
    foreach ($proto in $ProtoFiles) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($proto.Name)
        $messageModule = Join-Path $ProtoDir "$($stem)_pb2.py"
        $serviceModule = Join-Path $ProtoDir "$($stem)_pb2_grpc.py"

        if ($Clean) {
            foreach ($stale in @($messageModule, $serviceModule)) {
                if (Test-Path $stale) { Remove-Item $stale -Force }
            }
        }

        # Relative path on purpose: protoc resolves it against -I. and the
        # resulting descriptor name is what the generated imports are built
        # from. An absolute path here would produce unimportable stubs.
        $relative = "proto/$($proto.Name)"
        Write-Host "Generating  : $relative"

        & $PythonExe -m grpc_tools.protoc `
            -I. `
            --python_out=. `
            --grpc_python_out=. `
            $relative

        if ($LASTEXITCODE -ne 0) {
            throw "protoc failed for $relative (exit code $LASTEXITCODE)"
        }

        foreach ($expected in @($messageModule, $serviceModule)) {
            if (-not (Test-Path $expected)) {
                throw "Expected generated file was not produced: $expected"
            }
        }
    }

    # No proto/__init__.py is created: "from proto import client_pb2" already
    # resolves because proto/ is a PEP 420 namespace package. Add one only if
    # a top-level "proto" package ever appears on sys.path from a dependency
    # (proto-plus ships one), or if the project is packaged with setuptools
    # find_packages(), which skips directories without __init__.py.

    # Import the stubs through the same path client code will use.
    & $PythonExe -c "from proto import client_pb2, client_pb2_grpc; print('Import check: OK -', client_pb2.DESCRIPTOR.name)"
    if ($LASTEXITCODE -ne 0) {
        throw "Generated stubs could not be imported."
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Protobuf generation complete."
