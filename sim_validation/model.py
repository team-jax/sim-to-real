"""Load URDF geometry without changing the source model."""
from pathlib import Path
import xml.etree.ElementTree as ET
import tempfile

import mujoco

DEFAULT_URDF = Path(__file__).resolve().parents[1] / "models/JAXV2_Assembly_description/urdf/JAXV2_Assembly.urdf"


def load_robot(path=DEFAULT_URDF, visuals=False, environment=None):
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    if not visuals:
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                link.remove(visual)
    # Keep visual and contact geometry in separate viewer groups.
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for index, element in enumerate(link.findall(kind)):
                element.set("name", f"{link.get('name')}_{kind}_{index}")
    assets = {}
    for mesh in root.findall(".//mesh"):
        uri = mesh.attrib["filename"]
        relative = uri.removeprefix("package://").removeprefix("file://")
        candidates = [path.parent / relative,
                      path.parent.parent.parent / relative,
                      path.parent / Path(relative).name]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError(uri)
        name = source.name
        payload = source.read_bytes()
        if source.suffix.lower() == ".dae":
            import trimesh
            import numpy as np
            dae = ET.parse(source).getroot()
            ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
            unit = dae.find("c:asset/c:unit", ns)
            up = dae.find("c:asset/c:up_axis", ns)
            # This exporter uses meters and Z_UP; reject others rather than guess.
            if unit is None or float(unit.get("meter")) != 1 or up is None or up.text != "Z_UP":
                raise ValueError(f"Unsupported DAE coordinate convention: {source}")
            scene = trimesh.load(source, force="scene", process=False)
            geometry = scene.to_geometry()
            if not np.isfinite(geometry.vertices).all():
                raise ValueError(f"Nonfinite visual vertices: {source}")
            name = source.stem + "_visual.obj"
            payload = trimesh.exchange.obj.export_obj(geometry, include_color=False,
                                                       include_texture=False).encode()
        if name in assets and assets[name] != payload:
            raise ValueError(f"Conflicting mesh basename: {name}")
        assets[name] = payload
        mesh.set("filename", name)
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual=str(not visuals).lower(), balanceinertia="false")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    if environment is not None:
        from .scene import add_environment
        with tempfile.TemporaryDirectory() as directory:
            exported = Path(directory) / "robot.xml"
            mujoco.mj_saveLastXML(str(exported), model)
            mjcf = ET.parse(exported).getroot()
        add_environment(mjcf, **environment)
        model = mujoco.MjModel.from_xml_string(ET.tostring(mjcf, encoding="unicode"), assets)
    model.opt.timestep = 1 / 120
    # Camera-following illumination for inspecting the dark URDF materials.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.85
    model.vis.headlight.specular[:] = 0.2
    return model


def contact_pairs(model, data):
    """Return contacts including penetration depth; do not silently ignore them."""
    pairs = []
    for contact in data.contact:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[g])) for g in contact.geom]
        pairs.append((names[0], names[1], float(contact.dist)))
    return pairs
