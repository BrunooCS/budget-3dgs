#!/usr/bin/env bash
# Download the evaluation scenes into ${1:-data}:
#   tandt/{truck,train}, db/{drjohnson,playroom}   COLMAP releases of the 3DGS authors (~650 MB)
#   360_v2/{bicycle,garden,stump,room,counter,kitchen,bonsai}   Mip-NeRF 360 (~12.5 GB)
set -euo pipefail
root=${1:-data}
mkdir -p "$root" && cd "$root"

curl -L -o tandt_db.zip https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip -q tandt_db.zip && rm tandt_db.zip

curl -L -o 360_v2.zip http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
mkdir -p 360_v2 && unzip -q 360_v2.zip -d 360_v2 && rm 360_v2.zip
