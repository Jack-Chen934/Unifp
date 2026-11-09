"""Generate a disposable smooth-contact control without touching source assets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source_urdf = args.source / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
tree = ET.parse(source_urdf)
# Absolute mesh references let the one-file diagnostic reuse untouched geometry.
for mesh in tree.getroot().iter('mesh'):
    mesh.set('filename', str((source_urdf.parent / mesh.get('filename')).resolve()))
wheel_names = {corner + '_Link' for corner in ('fl', 'fr', 'bl', 'br')}
replaced = []
for link in tree.getroot().findall('link'):
    if link.get('name') not in wheel_names:
        continue
    for collision in link.findall('collision'):
        geometry = collision.find('geometry')
        mesh = geometry.find('mesh')
        if mesh is not None and mesh.get('filename').endswith('_solid_4.glb'):
            geometry.remove(mesh)
            ET.SubElement(geometry, 'sphere', {'radius': '0.1016'})
            replaced.append(link.get('name'))
assert len(replaced) == 4
args.output.joinpath('urdf').mkdir(parents=True, exist_ok=True)
target = args.output / 'urdf/a1arm_rgm3_desktop_fixed_gripper.urdf'
tree.write(target, encoding='utf-8', xml_declaration=True)
shutil.copyfile(args.source / 'asset_manifest.json', args.output / 'asset_manifest.json')
print(json.dumps({'diagnostic_only': True, 'source': str(source_urdf), 'candidate': str(target),
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'replaced_wheels': replaced,
                  'note_zh': '只作关闭自碰撞条件下的轮地接触隔离；不适用于正式训练。'}, ensure_ascii=False))
