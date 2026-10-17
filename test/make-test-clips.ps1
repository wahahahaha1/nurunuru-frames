# Generates synthetic 60 fps test clips (VP9/WebM) for the interpolation quality harness.
# Uses the ffmpeg bundled with SVP 4. Output: test\clips\*.webm
$ErrorActionPreference = 'Stop'
$ff = 'C:\Program Files (x86)\SVP 4\utils\ffmpeg.exe'
$dir = Join-Path $PSScriptRoot 'clips'
New-Item -ItemType Directory -Force $dir | Out-Null
$bg = Join-Path $dir 'bg.png'
# short GOP so the harness can seek frame-by-frame quickly
$vp9 = @('-c:v', 'libvpx-vp9', '-crf', '12', '-b:v', '0', '-deadline', 'good', '-cpu-used', '4', '-row-mt', '1', '-g', '8', '-pix_fmt', 'yuv420p', '-an')

function Run([string[]]$a) {
    & $ff -hide_banner -loglevel error -y @a
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed: $a" }
}

# Detailed colourful background: mandelbrot + perlin noise luminance texture
Run @('-f', 'lavfi', '-i', 'mandelbrot=s=3840x1440:start_scale=0.6:start_x=-0.743:start_y=0.131:maxiter=2000',
      '-f', 'lavfi', '-i', 'perlin=s=3840x1440:octaves=8:persistence=0.75:xscale=10:yscale=10',
      '-filter_complex', '[1:v]format=rgb24[n];[0:v][n]blend=all_mode=overlay:all_opacity=0.9,format=rgb24',
      '-frames:v', '1', $bg)

# 1. Medium pan: 12 px per 60fps frame horizontally, gentle vertical sway
Run (@('-loop', '1', '-framerate', '60', '-i', $bg, '-t', '3',
       '-vf', "crop=1280:720:x='t*720':y='300+80*sin(t*2)'") + $vp9 + @((Join-Path $dir 'pan.webm')))

# 2. Fast pan: ~21 px per 60fps frame
Run (@('-loop', '1', '-framerate', '60', '-i', $bg, '-t', '2',
       '-vf', "crop=1280:720:x='t*1270':y='200'") + $vp9 + @((Join-Path $dir 'fastpan.webm')))

# 3. Moving object over a slowly panning background (occlusion test)
Run (@('-loop', '1', '-framerate', '60', '-i', $bg, '-f', 'lavfi', '-i', 'testsrc2=s=360x240:r=60', '-t', '3',
       '-filter_complex', "[0:v]crop=1280:720:x='200+t*120':y=400[b];[b][1:v]overlay=x='60+t*330':y='240+100*sin(t*3)':shortest=1") + $vp9 + @((Join-Path $dir 'object.webm')))

# 4. Zoom in
Run (@('-loop', '1', '-framerate', '60', '-i', $bg, '-t', '3',
       '-vf', "scale=2560:960,zoompan=z='1+0.004*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=60") + $vp9 + @((Join-Path $dir 'zoom.webm')))

# 5. testsrc2 (moving shapes, text, gradients)
Run (@('-f', 'lavfi', '-i', 'testsrc2=s=1280x720:r=60', '-t', '3') + $vp9 + @((Join-Path $dir 'testsrc.webm')))

Get-ChildItem $dir | Select-Object Name, Length

