"""Deployment-reference placement, with original Isaac self-contact policy."""
import itertools
import math
import xml.etree.ElementTree as ET

LAYOUTS = {
    "reference": (0.740, -0.225),
    "center10": (0.735, -0.100),
    # base_link DAE spans local y=+-0.125 m; yaw -90 maps local y to world x.
    "edge10": (0.735, -0.125-0.100),
}
CUBE_SIDE_M = 0.04
CUBE_HALF_M = CUBE_SIDE_M / 2
CUBE_MASS_KG = 0.05
CUBE_FRICTION = 0.4
CUBE_DYNAMIC_FRICTION = 0.35
TABLE_STATIC_FRICTION = 1.2
TABLE_DYNAMIC_FRICTION = 1.0
# MuJoCo has one sliding coefficient. Match the original sliding regime;
# static breakaway cannot simultaneously match the separate PhysX value.
CONTACT_FRICTION_BASIS = "PhysX dynamic coefficients, arithmetic pair mean"


def add_environment(root, dx=0.0, dy=0.0, self_collision=False, layout="reference"):
    if not all(math.isfinite(v) for v in (dx, dy)):
        raise ValueError("Cube offsets must be finite")
    height, table_edge = LAYOUTS[layout]
    cube_x = table_edge - 0.06
    cube_z = height + CUBE_HALF_M
    world = root.find("worldbody")
    robot_bodies = [b.get("name") for b in world.iter("body")]
    placement = ET.Element("body", name="robot_mount", pos="0 0 0.305",
                           quat="0.7071067811865476 0 0 -0.7071067811865476")
    for element in list(world):
        world.remove(element)
        placement.append(element)
    world.append(placement)
    # URDF does not encode Isaac materials. Use dynamic values, and define
    # explicit pairs below to avoid MuJoCo's default max combination.
    robot_geoms = []
    for geom in placement.iter("geom"):
        if geom.get("contype", "1") != "0":
            geom.set("friction", f"{CUBE_DYNAMIC_FRICTION} 0.005 0.0001")
            robot_geoms.append(geom.get("name"))
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1",
                  rgba="0.22 0.25 0.29 1", group="2")
    ET.SubElement(world, "geom", name="table", type="box", pos=f"{table_edge-0.4} 0 {height-0.02}",
                  size="0.4 0.35 0.02", rgba="0.52 0.43 0.32 1", friction=f"{TABLE_DYNAMIC_FRICTION} 0.005 0.0001", group="2")
    cube = ET.SubElement(world, "body", name="cube", pos=f"{cube_x+dx} {0.26+dy} {cube_z}")
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(cube, "geom", name="cube_geom", type="box", size=" ".join([str(CUBE_HALF_M)]*3),
                  mass=str(CUBE_MASS_KG), rgba="0.85 0.24 0.12 1", friction=f"{CUBE_DYNAMIC_FRICTION} 0.005 0.0001", group="2")
    # Above the cube so the reference marker remains visible at zero offset.
    ET.SubElement(world, "site", name="placement_0_0", type="sphere", pos=f"{cube_x} 0.26 {height+0.10}",
                  size="0.008", rgba="0.1 0.95 0.3 1", group="3")
    # A horizontal ruler shows exactly which endpoints define the 10 cm gap.
    if layout != "reference":
        start = 0 if layout == "center10" else -0.125
        ET.SubElement(world, "site", name="gap_10cm", type="capsule",
                      fromto=f"{start} 0 {height+0.015} {table_edge} 0 {height+0.015}",
                      size="0.003", rgba="1 0.85 0.1 1", group="3")
    for name, end, color in [("x", "0.1 0 0.003", "1 0.2 0.2 1"),
                             ("y", "0 0.1 0.003", "0.2 1 0.2 1"),
                             ("z", "0 0 0.1", "0.2 0.4 1 1")]:
        ET.SubElement(world, "site", name=f"world_{name}", type="capsule", fromto="0 0 0.003 "+end,
                      size="0.003", rgba=color, group="3")
    if not self_collision:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for a, b in itertools.combinations(robot_bodies, 2):
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    def pair(a, b, mu):
        ET.SubElement(contact, "pair", geom1=a, geom2=b, condim="3",
                      friction=f"{mu} {mu} 0.005 0.0001 0.0001")
    table_contact_mu = (TABLE_DYNAMIC_FRICTION + CUBE_DYNAMIC_FRICTION) / 2
    pair("table", "cube_geom", table_contact_mu)
    for geom in robot_geoms:
        pair(geom, "cube_geom", CUBE_DYNAMIC_FRICTION)
        pair(geom, "table", table_contact_mu)
    actuator = ET.SubElement(root, "actuator")
    for joint in placement.iter("joint"):
        name = joint.get("name")
        fingers = "finger" in name
        if fingers:
            joint.set("armature", "0.0002")
        ET.SubElement(actuator, "position", name=f"motor_{name}", joint=name,
                      kp="40" if fingers else "100", kv="2" if fingers else "8",
                      forcelimited="true", forcerange="-100 100")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", "implicitfast")
