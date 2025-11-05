# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert all URDF files in a directory into USD format.

This script recursively finds all URDF files in the given input directory and converts each one
into a corresponding USD file using the Isaac Sim URDF importer.

positional arguments:
  input               The path to the input directory containing URDF files.
  output              The path to the output directory where USD files will be stored.
                      (Keeps the same relative structure as input)

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --joint-stiffness         The stiffness of the joint drive. (default: 100.0)
  --joint-damping           The damping of the joint drive. (default: 1.0)
  --joint-target-type       The type of control to use for the joint drive. (default: "position")
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert all URDF files in a directory into USD format.")
parser.add_argument("input", type=str, help="The path to the input directory containing URDF files.")
parser.add_argument("output", type=str, help="The path to the output directory for USD files.")
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the base to where it is imported.")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="The type of control to use for the joint drive.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def convert_single_urdf(urdf_path: str, dest_usd_path: str):
    """Convert a single URDF file to USD."""
    # Check valid file path
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        print(f"[WARNING] Invalid file path: {urdf_path}")
        return False

    # Create destination directory
    dest_dir = os.path.dirname(dest_usd_path)
    os.makedirs(dest_dir, exist_ok=True)

    # Create Urdf converter config
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=dest_dir,
        usd_file_name=os.path.basename(dest_usd_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    # Print info
    print("-" * 80)
    print(f"Converting: {urdf_path}")
    print(f"      ->   {dest_usd_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)

    # Create Urdf converter and import the file
    try:
        urdf_converter = UrdfConverter(urdf_converter_cfg)
        print("URDF importer output:")
        print(f"Generated USD file: {urdf_converter.usd_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert {urdf_path}: {e}")
        return False

    return True


def main():
    input_dir = args_cli.input
    output_dir = args_cli.output

    # Validate input directory
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input is not a valid directory: {input_dir}")

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    # Walk through all .urdf files
    urdf_files = Path(input_dir).rglob("*.urdf")
    converted_count = 0

    for urdf_path in urdf_files:
        urdf_path = str(urdf_path)

        # Compute relative path from input_dir
        rel_path = os.path.relpath(urdf_path, input_dir)
        usd_filename = os.path.splitext(os.path.basename(rel_path))[0] + ".usd"
        dest_usd_path = os.path.join(output_dir, os.path.dirname(rel_path), usd_filename)

        success = convert_single_urdf(urdf_path, dest_usd_path)
        if success:
            converted_count += 1

    print(f"✅ Batch conversion completed. {converted_count} files converted.")
    print(f"📁 Output directory: {output_dir}")

    # Determine if there is a GUI to update
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Simulate scene (if not headless)
    if local_gui or livestream_gui:
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Script failed: {e}")
    finally:
        simulation_app.close()