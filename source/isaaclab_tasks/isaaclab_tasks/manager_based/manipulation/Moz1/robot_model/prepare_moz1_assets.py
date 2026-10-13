#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download the public Moz1 URDF package and build the Pink-WBC URDF.

The vendor URDF models the four mecanum wheels as independent joints.  Wheel
angles do not change the base pose in a URDF kinematic tree, so they cannot be
used directly by a differential IK solver.  The adapted model fixes the wheel
joints and inserts three virtual joints (x, y and yaw) in front of ``base_link``.
This is the same planar-base abstraction used by the TianJi whole-body demo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

OFFICIAL_ARCHIVE_URL = "https://docs.spirit-ai.com/package/spirit01_model.zip"
OFFICIAL_ARCHIVE_SHA256 = "f5426944b2ba7b92292d12a6cb1a807a4161b0b93f4473ecc1cf7625053569dc"
OFFICIAL_URDF_SHA256 = "17e11149b09507530ac177e7708b779384deab04e0a8184fbfe515d4712e5fd5"

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = THIS_DIR / "spirit01_model"
ORIGINAL_URDF = PACKAGE_DIR / "urdf" / "moz1_original.urdf"
WBC_URDF = PACKAGE_DIR / "urdf" / "moz1_wbc.urdf"

# Speeds are taken from the public Moz1 mechanical-structure table (degrees/s).
JOINT_VELOCITY_RAD_S = {
    "LegWaist_0": 1.134464,
    "LegWaist_1": 1.134464,
    "LegWaist_2": 1.134464,
    "LegWaist_3": 1.134464,
    "LegWaist_4": 1.134464,
    "LegWaist_5": 1.134464,
    "LeftArm_0": 3.665191,
    "LeftArm_1": 3.665191,
    "LeftArm_2": 4.188790,
    "LeftArm_3": 4.188790,
    "LeftArm_4": 5.235988,
    "LeftArm_5": 5.235988,
    "LeftArm_6": 5.235988,
    "RightArm_0": 3.665191,
    "RightArm_1": 3.665191,
    "RightArm_2": 4.188790,
    "RightArm_3": 4.188790,
    "RightArm_4": 5.235988,
    "RightArm_5": 5.235988,
    "RightArm_6": 5.235988,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_package() -> None:
    """Download and unpack the public vendor package into this directory."""
    with tempfile.TemporaryDirectory(prefix="moz1-assets-") as temp_dir:
        archive_path = Path(temp_dir) / "spirit01_model.zip"
        urllib.request.urlretrieve(OFFICIAL_ARCHIVE_URL, archive_path)
        digest = _sha256(archive_path)
        if digest != OFFICIAL_ARCHIVE_SHA256:
            raise RuntimeError(f"Unexpected archive SHA-256: {digest}")

        with zipfile.ZipFile(archive_path) as archive:
            members = [
                name for name in archive.namelist() if name.startswith("spirit01_model/") and not name.endswith("/")
            ]
            for member in members:
                relative = Path(member).relative_to("spirit01_model")
                if relative.parts[0] not in {"meshes", "urdf"} and relative.name not in {
                    "README.md",
                    "package.xml",
                }:
                    continue
                output = PACKAGE_DIR / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    downloaded_urdf = PACKAGE_DIR / "urdf" / "moz1.urdf"
    if _sha256(downloaded_urdf) != OFFICIAL_URDF_SHA256:
        raise RuntimeError("The downloaded Moz1 URDF does not match the expected SHA-256.")
    downloaded_urdf.replace(ORIGINAL_URDF)
    official_readme = PACKAGE_DIR / "README.md"
    if official_readme.exists():
        official_readme.replace(PACKAGE_DIR / "README.official.md")


def _virtual_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-6", "ixy": "0", "ixz": "0", "iyy": "1e-6", "iyz": "0", "izz": "1e-6"},
    )
    return link


def _virtual_joint(
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    axis: str,
    lower: float,
    upper: float,
    velocity: float,
) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(
        joint,
        "limit",
        {"lower": str(lower), "upper": str(upper), "effort": "5000", "velocity": str(velocity)},
    )
    ET.SubElement(joint, "dynamics", {"damping": "10", "friction": "0"})
    return joint


def build_wbc_urdf() -> None:
    """Create a deterministic planar-base WBC model from the vendor URDF."""
    if not ORIGINAL_URDF.exists():
        raise FileNotFoundError(f"Missing {ORIGINAL_URDF}; run this script with --download first.")

    tree = ET.parse(ORIGINAL_URDF)
    robot = tree.getroot()
    robot.set("name", "MOZ1_WBC")

    for joint in robot.findall("joint"):
        # Hyphens are sanitized by USD.  Rename at the URDF source so Isaac Lab
        # and Pinocchio see exactly the same joint names.
        joint_name = joint.attrib["name"].replace("-", "_")
        joint.set("name", joint_name)

        if joint_name.startswith("Base_"):
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                child = joint.find(tag)
                if child is not None:
                    joint.remove(child)
            continue

        limit = joint.find("limit")
        if limit is not None and joint_name in JOINT_VELOCITY_RAD_S:
            # The public URDF has zeros in these fields.  Public speed limits
            # are restored here; the public package does not provide torques.
            limit.set("velocity", str(JOINT_VELOCITY_RAD_S[joint_name]))
            limit.set("effort", "1000")
            if joint.find("dynamics") is None:
                ET.SubElement(joint, "dynamics", {"damping": "1", "friction": "0"})

    virtual_elements = [
        _virtual_link("world_base"),
        _virtual_link("base_x_link"),
        _virtual_link("base_y_link"),
        _virtual_joint("dummy_x_joint", "prismatic", "world_base", "base_x_link", "1 0 0", -2.0, 2.0, 1.0),
        _virtual_joint("dummy_y_joint", "prismatic", "base_x_link", "base_y_link", "0 1 0", -2.0, 2.0, 1.0),
        _virtual_joint(
            "dummy_yaw_joint", "revolute", "base_y_link", "base_link", "0 0 1", -3.141592654, 3.141592654, 1.5
        ),
    ]
    for index, element in enumerate(virtual_elements):
        robot.insert(index, element)

    ET.indent(tree, space="  ")
    WBC_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(WBC_URDF, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Redownload the public vendor package first.")
    args = parser.parse_args()

    if args.download:
        download_official_package()
    build_wbc_urdf()
    print(f"Generated {WBC_URDF}")


if __name__ == "__main__":
    main()
