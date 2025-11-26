from pathlib import Path

from dex_retargeting import yourdfpy as urdf

urdf_path = Path('assets/Shadow/shadow_hand_right_woarm.urdf')
output_path = Path('assets/Shadow/shadow_hand_right_woarm_dummyjoint.urdf')
robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
urdf_name = urdf_path.name
robot_urdf.write_xml_file(output_path)