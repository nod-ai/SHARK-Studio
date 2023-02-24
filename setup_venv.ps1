<#
.SYNOPSIS
  A script to update and install the SHARK runtime and its dependencies.

.DESCRIPTION
  This script updates and installs the SHARK runtime and its dependencies.
  It checks the Python version installed and installs any required build
  dependencies into a Python virtual environment.
  If that environment does not exist, it creates it.
  
.PARAMETER update-src
  git pulls latest version

.PARAMETER force
  removes and recreates venv to force update of all dependencies
  
.EXAMPLE
  .\setup_venv.ps1 --force

.EXAMPLE
  .\setup_venv.ps1 --update-src

.INPUTS
  None

.OUTPUTS
  None

#>

param([string]$arguments)

if ($arguments -eq "--update-src"){
	git pull
}

if ($arguments -eq "--force"){
	if (Test-Path env:VIRTUAL_ENV) {
        Write-Host "deactivating..."
        Deactivate
    }
    
    if (Test-Path .\shark.venv\) {
        Write-Host "removing and recreating venv..."
        Remove-Item .\shark.venv -Force -Recurse
        if (Test-Path .\shark.venv\) {
            Write-Host 'could not remove .\shark-venv - please try running ".\setup_venv.ps1 --force" again!'
            break
        }
    }
}

# redirect stderr into stdout
$p = &{python -V} 2>&1
# check if an ErrorRecord was returned
$version = if($p -is [System.Management.Automation.ErrorRecord])
{
    # grab the version string from the error message
    $p.Exception.Message
}
else
{
    # otherwise return complete Python list
    $ErrorActionPreference = 'SilentlyContinue'
    $PyVer = py --list
}

# deactivate any activated venvs
if ($PyVer -like "*venv*")
{
  deactivate # make sure we don't update the wrong venv
  $PyVer = py --list # update list
}

Write-Host "Python versions found are"
Write-Host ($PyVer | Out-String) # formatted output with line breaks
if (!($PyVer.length -ne 0)) {$p} # return Python --version String if py.exe is unavailable
if (!($PyVer -like "*3.11*") -and !($p -like "*3.11*")) # if 3.11 is not in any list
{
    Write-Host "Please install Python 3.11 and try again"
    break
}

Write-Host "Installing Build Dependencies"
# make sure we really use 3.11 from list, even if it's not the default.
if (!($PyVer.length -ne 0)) {py -3.11 -m venv .\shark.venv\}
else {python -m venv .\shark.venv\}
.\shark.venv\Scripts\activate
python -m pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
pip install --pre torch-mlir torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu -f https://llvm.github.io/torch-mlir/package-index/
pip install --upgrade -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html iree-compiler iree-runtime
Write-Host "Building SHARK..."
pip install -e . -f https://llvm.github.io/torch-mlir/package-index/ -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html
Write-Host "Build and installation completed successfully"
Write-Host "Source your venv with ./shark.venv/Scripts/activate"
