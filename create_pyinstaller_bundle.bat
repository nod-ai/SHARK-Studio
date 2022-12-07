pyinstaller.exe web/shark_sd.spec --clean -y --distpath ./temp/dist --workpath ./temp/build
tufup targets add %1 temp/dist temp/keystore
