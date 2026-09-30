$ErrorActionPreference = "Stop"

$parent = "D:\BaiduNetdiskDownload\FLIR3C"
$source = "D:\BaiduNetdiskDownload\FLIR_mm"
$target = Join-Path $parent "FLIR3C"
$staging = Join-Path $parent "FLIR3C_swap350_staging"
$backup = Join-Path $parent "FLIR3C_random794_backup_20260922"
$manifestDir = "D:\BaiduNetdiskDownload\MutilModel_3398475911\datasets\flir_custom_swap350"

$resolvedParent = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $parent).Path).TrimEnd("\")
foreach ($path in @($target, $staging, $backup)) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($resolvedParent + "\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe path outside intended FLIR3C parent: $full"
    }
}
if (-not (Test-Path -LiteralPath $source)) { throw "Official source not found: $source" }
if (-not (Test-Path -LiteralPath $target)) { throw "Current FLIR3C target not found: $target" }
if (Test-Path -LiteralPath $staging) { throw "Staging path already exists: $staging" }
if (Test-Path -LiteralPath $backup) { throw "Backup path already exists: $backup" }

$trainToTest = @(Get-Content -LiteralPath (Join-Path $manifestDir "official_train_to_custom_test_350.txt") | Where-Object { $_.Trim() })
$testToTrain = @(Get-Content -LiteralPath (Join-Path $manifestDir "official_test_to_custom_train_350.txt") | Where-Object { $_.Trim() })
if ($trainToTest.Count -ne 350 -or $testToTrain.Count -ne 350) {
    throw "Expected 350 names in each exchange list, got $($trainToTest.Count)/$($testToTrain.Count)"
}

foreach ($rel in @("images\train", "images\test", "images_ir\train", "images_ir\test", "labels\train", "labels\test")) {
    New-Item -ItemType Directory -Path (Join-Path $staging $rel) -Force | Out-Null
}

Write-Host "[1/5] Copying official RGB, IR and labels to staging..."
foreach ($modality in @("images", "images_ir")) {
    foreach ($split in @("train", "test")) {
        $srcDir = Join-Path $source "$modality\$split"
        $dstDir = Join-Path $staging "$modality\$split"
        Get-ChildItem -LiteralPath $srcDir -File -Filter "*.jpg" | Copy-Item -Destination $dstDir
    }
}
foreach ($split in @("train", "test")) {
    $srcDir = Join-Path $source "labels\$split"
    $dstDir = Join-Path $staging "labels\$split"
    Get-ChildItem -LiteralPath $srcDir -File -Filter "*.txt" | Copy-Item -Destination $dstDir
}

function Move-Sample([string]$name, [string]$fromSplit, [string]$toSplit) {
    foreach ($modality in @("images", "images_ir")) {
        $src = Join-Path $staging "$modality\$fromSplit\$name"
        $dst = Join-Path $staging "$modality\$toSplit\$name"
        if (-not (Test-Path -LiteralPath $src)) { throw "Missing source file: $src" }
        Move-Item -LiteralPath $src -Destination $dst
    }
    $labelName = [IO.Path]::ChangeExtension($name, ".txt")
    $labelSrc = Join-Path $staging "labels\$fromSplit\$labelName"
    $labelDst = Join-Path $staging "labels\$toSplit\$labelName"
    if (-not (Test-Path -LiteralPath $labelSrc)) { throw "Missing label: $labelSrc" }
    Move-Item -LiteralPath $labelSrc -Destination $labelDst
}

Write-Host "[2/5] Exchanging exactly 350 samples in each direction..."
foreach ($name in $trainToTest) { Move-Sample $name "train" "test" }
foreach ($name in $testToTrain) { Move-Sample $name "test" "train" }

$dataYaml = @"
# FLIR RGB+IR custom split: exactly 350 official-train images exchanged with 350 official-test images.
# This is a custom experimental split, not the official FLIR evaluation split.
path: D:/BaiduNetdiskDownload/FLIR3C/FLIR3C
train: ./images/train
val: ./images/test
test: ./images/test

modality:
  rgb: images
  ir: images_ir

modality_used: ['rgb', 'ir']
Xch: 3
nc: 3
names:
  0: person
  1: car
  2: bicycle
"@
[IO.File]::WriteAllText((Join-Path $staging "data.yaml"), $dataYaml.TrimStart(), [Text.UTF8Encoding]::new($false))
New-Item -ItemType Directory -Path (Join-Path $staging "split_manifest") -Force | Out-Null
foreach ($name in @("manifest.txt", "official_train_to_custom_test_350.txt", "official_test_to_custom_train_350.txt", "train_names.txt", "test_names.txt")) {
    Copy-Item -LiteralPath (Join-Path $manifestDir $name) -Destination (Join-Path $staging "split_manifest\$name")
}

Write-Host "[3/5] Verifying staged dataset..."
$expected = @{
    "images\train" = 4129; "images\test" = 1013
    "images_ir\train" = 4129; "images_ir\test" = 1013
    "labels\train" = 4129; "labels\test" = 1013
}
foreach ($rel in $expected.Keys) {
    $count = @(Get-ChildItem -LiteralPath (Join-Path $staging $rel) -File).Count
    if ($count -ne $expected[$rel]) { throw "Count mismatch for $rel`: $count, expected $($expected[$rel])" }
}
foreach ($split in @("train", "test")) {
    $rgbNames = @(Get-ChildItem -LiteralPath (Join-Path $staging "images\$split") -File | ForEach-Object BaseName | Sort-Object)
    $irNames = @(Get-ChildItem -LiteralPath (Join-Path $staging "images_ir\$split") -File | ForEach-Object BaseName | Sort-Object)
    $labelNames = @(Get-ChildItem -LiteralPath (Join-Path $staging "labels\$split") -File | ForEach-Object BaseName | Sort-Object)
    if (Compare-Object $rgbNames $irNames) { throw "RGB/IR mismatch in $split" }
    if (Compare-Object $rgbNames $labelNames) { throw "RGB/label mismatch in $split" }
}

Write-Host "[4/5] Preserving existing random-794 dataset as backup..."
Move-Item -LiteralPath $target -Destination $backup
try {
    Write-Host "[5/5] Activating custom swap-350 dataset..."
    Move-Item -LiteralPath $staging -Destination $target
}
catch {
    if ((Test-Path -LiteralPath $backup) -and -not (Test-Path -LiteralPath $target)) {
        Move-Item -LiteralPath $backup -Destination $target
    }
    throw
}

Write-Host "DONE"
Write-Host "Active: $target"
Write-Host "Backup: $backup"
Write-Host "Train/Test: 4129/1013"
Write-Host "Official train->custom test: 350"
Write-Host "Official test->custom train: 350"