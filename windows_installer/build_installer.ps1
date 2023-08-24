param (
    # Advanced Installer path in the computer
    [Parameter(Mandatory=$true)]
    [string]$ai_path,
    # Version of MSI build
    [Parameter(Mandatory=$true)]
    [string]$version,
    # Flag to indicate beta version
    [switch]$beta
)

# Useful constants
$project_path = "./windows_installer/shark_studio.aip"
$updater_path = "./windows_installer/shark_updater.aip"
$project_files_path = -join($(pwd),"\dist\studio_bundle")

# These variables are affected by beta flag
$update_info_url = "https://github.com/nod-ai/SHARK/releases/latest/download/shark_updater.txt"
$update_msi_url = "https://github.com/nod-ai/SHARK/releases/latest/download/shark_studio.msi"
$msi_path = ".\windows_installer\shark_studio-SetupFiles\shark_studio.msi"

if ($beta) {
    $update_info_url = "https://github.com/nod-ai/SHARK/releases/download/latest-nightly/shark_updater_beta.txt"
    $update_msi_url = "https://github.com/nod-ai/SHARK/releases/download/latest-nightly/shark_studio_beta.msi"
    $msi_path = ".\windows_installer\shark_studio-SetupFiles\shark_studio_beta.msi"
}


# Assuming pyinstaller has already been run

# Configures AIP to latest version of ./dist/studio_bundle
Write-Host "removing old files and shortcuts"
&$ai_path /edit $project_path /DelShortcut -name "Shark Studio" -dir "SHORTCUTDIR"
&$ai_path /edit $project_path /DelFolder "APPDIR\studio_bundle"

Write-Host "adding new files and shortcuts"
&$ai_path /edit $project_path /AddFolder "APPDIR" $project_files_path
&$ai_path /edit $project_path /NewShortcut -name "Shark Studio" -target "APPDIR\studio_bundle\studio_bundle.exe" -dir "SHORTCUTDIR" -runasadmin -pin_to_taskbar

Write-Host "setting version to $version"
&$ai_path /edit $project_path /SetVersion $version

# Sets update url based on beta flag
&$ai_path /edit $project_path /SetUpdatesUrl $update_info_url

# Builds the msi
&$ai_path /rebuild $project_path

# Renames the msi if beta flag is on (TODO)
if ($beta) {
    Rename-Item -Path "./windows_installer/shark_studio-SetupFiles/shark_studio.msi" -NewName "shark_studio_beta.msi"
}

# Resets to latest msi
&$ai_path /edit $updater_path /DeleteUpdate Update
&$ai_path /edit $updater_path /NewUpdate $msi_path -name Update -display_name "shark_update" -url $update_msi_url

# builds shark_updater.txt
&$ai_path /rebuild $updater_path

# Renames shark_updater.txt if beta flag is on
if ($beta) {
    Rename-Item -Path "./windows_installer/shark_updater-SetupFiles/shark_updater.txt" -NewName "shark_updater_beta.txt"
}
