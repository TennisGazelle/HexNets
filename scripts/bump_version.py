import argparse
import os
import sys

hexnet_version_line = ""
version = {}

parser = argparse.ArgumentParser(
    prog="Versioning", description="Updates the version", epilog="Tennis Gazelle Productions"
)

parser.add_argument("--major", action="store_true")  # on/off flag
parser.add_argument("--minor", action="store_true")  # on/off flag
parser.add_argument("--patch", action="store_true")  # on/off flag
parser.add_argument("filename")

args = parser.parse_args()

# load
with open(os.path.abspath(args.filename), "r") as version_file:
    lines = version_file.readlines()

for l in lines:
    if l.startswith("version"):
        hexnet_version_line = l
        version_word = l.split("=")[1].strip().replace('"', "").split(".")
        version["major"] = int(version_word[0])
        version["minor"] = int(version_word[1])
        version["patch"] = int(version_word[2])
        break
else:
    print("Version not found in file", file=sys.stderr)
    exit(1)

if args.major:
    version["patch"] = 0
    version["minor"] = 0
    version["major"] += 1
if args.minor:
    version["patch"] = 0
    version["minor"] += 1
if args.patch:
    version["patch"] += 1
new_version = f'{version["major"]}.{version["minor"]}.{version["patch"]}'
print(new_version)

if args.major or args.minor or args.patch:
    lines[lines.index(hexnet_version_line)] = f'version = "{new_version}"\n'
    with open(os.path.abspath(args.filename), "w") as version_file:
        version_file.writelines(lines)
