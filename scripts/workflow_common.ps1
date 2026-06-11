function Get-ProjectPythonCommand {
    param([string]$ConfiguredLauncher)
    if ($ConfiguredLauncher -and $ConfiguredLauncher.Trim().Length -gt 0) {
        return $ConfiguredLauncher
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py -3"
    }
    return "python"
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [string]$Launcher = ""
    )
    $cmd = Get-ProjectPythonCommand -ConfiguredLauncher $Launcher
    if ($cmd -eq "py -3") {
        & py -3 @Arguments
    } elseif ($cmd -eq "python") {
        & python @Arguments
    } else {
        $parts = $cmd -split " "
        $exe = $parts[0]
        $prefixArgs = @()
        if ($parts.Count -gt 1) {
            $prefixArgs = $parts[1..($parts.Count - 1)]
        }
        & $exe @prefixArgs @Arguments
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Add-OptionalArg {
    param(
        [Parameter(Mandatory=$true)][System.Collections.Generic.List[string]]$Args,
        [Parameter(Mandatory=$true)][string]$Name,
        $Value
    )
    if ($null -ne $Value -and "$Value".Length -gt 0) {
        $Args.Add($Name) | Out-Null
        $Args.Add([string]$Value) | Out-Null
    }
}

function Assert-PathExists {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}
