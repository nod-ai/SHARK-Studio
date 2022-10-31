#Write-Host "Installing python"

#Start-Process winget install Python.Python.3.10 '/quiet InstallAllUsers=1 PrependPath=1' -wait -NoNewWindow

#Write-Host "python installation completed successfully"

#Write-Host "Reload environment variables"
#$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
#Write-Host "Reloaded environment variables"


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
    # otherwise return as is
    $p
}

Write-Host "Python version found is"
Write-Host $p


Write-Host "Installing Build Dependencies"
python -m venv .\shark.venv\
.\shark.venv\Scripts\activate
pip install -r requirements.txt
pip install --pre torch-mlir torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu116 -f https://llvm.github.io/torch-mlir/package-index/
pip install --upgrade -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html iree-compiler iree-runtime
Write-Host "Building SHARK..."
pip install -e . -f https://llvm.github.io/torch-mlir/package-index/ -f https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html
pip install diffusers transformers scipy
Write-Host "Build and installation completed successfully"
Write-Host "Source your venv with ./shark.venv/Scripts/activate"
