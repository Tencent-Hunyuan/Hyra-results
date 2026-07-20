#!/bin/bash
set -e
python3 -c "import json; json.dump({'build_script': open('build_hunyuan.py').read()}, open('solution.json','w'))"
