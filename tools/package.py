"""Builds the distributable zip: dist/nurunuru-frames-<version>.zip with manifest.json at the root
(usable for the Chrome Web Store, or unzip + "Load unpacked"). Run with: py -3.10 -E tools/package.py"""
import json, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "extension")
DIST = os.path.join(ROOT, "dist")

manifest = json.load(open(os.path.join(EXT, "manifest.json"), encoding="utf-8"))

# every file the manifest refers to must exist
refs = [manifest["background"]["service_worker"], manifest["action"]["default_popup"]]
refs += list(manifest["icons"].values()) + list(manifest["action"]["default_icon"].values())
for cs in manifest["content_scripts"]:
    refs += cs["js"]
missing = [r for r in refs if not os.path.isfile(os.path.join(EXT, r))]
if missing:
    sys.exit(f"missing files referenced by manifest.json: {missing}")

os.makedirs(DIST, exist_ok=True)
out = os.path.join(DIST, f"nurunuru-frames-{manifest['version']}.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for base, dirs, files in os.walk(EXT):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = os.path.join(base, name)
            z.write(path, os.path.relpath(path, EXT).replace(os.sep, "/"))
    z.write(os.path.join(ROOT, "LICENSE"), "LICENSE")
print(out)
with zipfile.ZipFile(out) as z:
    for i in z.infolist():
        print(f"  {i.file_size:>8}  {i.filename}")
