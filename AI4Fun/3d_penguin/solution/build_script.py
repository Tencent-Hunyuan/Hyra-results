import bpy, math, mathutils

# ----------------------------------------------------------------------------
# Tencent / QQ penguin -- one coherent solid body from smooth primitives.
# +Z up, faces -Y (front toward -Y). Solid colors only. No camera/lights.
#
# Starting from a strong baseline configuration, the geometry is tuned so the
# characteristic silhouette reads clearly from the front:
#   * SCARF: a FLATTER wrapped band (thin front-to-back) in a deep saturated
#     red, plus a LONG forked hanging tail flap dropping down the front over
#     the belly.
#   * BEAK: a larger, bright, clearly OPEN smiling bill.
#   * BODY: a slightly taller egg (narrower head over a wide body).
#   * FEET: broad, flat, forward-splayed webbed feet.
#   * Viewer-LEFT eye open (-X), viewer-RIGHT eye winking (+X).
# ----------------------------------------------------------------------------

def make_mat(name, rgb, rough=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    b.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.2
    return m

BLACK   = make_mat("black",   (0.027, 0.027, 0.030))
WHITE   = make_mat("white",   (0.975, 0.975, 0.975))
ORANGE  = make_mat("orange",  (1.00, 0.55, 0.02))
RED     = make_mat("red",     (0.74, 0.02, 0.02), rough=0.42)
DARKRED = make_mat("darkred", (0.34, 0.015, 0.015))

def add_ellipsoid(loc, scale, mat, name, rot=(0, 0, 0), seg=64, ring=36):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=ring, location=loc)
    o = bpy.context.object
    o.scale = scale
    o.rotation_euler = rot
    o.name = name
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o

def add_box(loc, scale, mat, name, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.object
    o.scale = scale
    o.rotation_euler = rot
    o.name = name
    o.data.materials.append(mat)
    return o

# === BODY: continuous egg -- slightly taller than wide, narrower head heavily
#     overlapping a wide squat lower body; neck filler hides the waist seam.
add_ellipsoid((0.0, 0.02, 0.82), (1.17, 1.12, 1.16), BLACK, "body")
add_ellipsoid((0.0, -0.03, 1.70), (0.99, 0.98, 0.97), BLACK, "head")
add_ellipsoid((0.0, 0.0, 1.30), (1.10, 1.06, 0.66), BLACK, "neck")

# === WHITE belly: one large smooth oval on the lower front, below the scarf.
add_ellipsoid((0.0, -0.42, 0.60), (0.95, 0.88, 1.06), WHITE, "belly")

# === EYES: two white ovals on the BLACK head, divided by black.
#   open eye -> viewer-LEFT (-X);  wink -> viewer-RIGHT (+X)
add_ellipsoid((-0.33, -0.86, 1.96), (0.235, 0.135, 0.305), WHITE, "eyeL_white")
add_ellipsoid((-0.33, -0.99, 1.94), (0.135, 0.095, 0.185), BLACK, "eyeL_pupil")
add_ellipsoid((-0.29, -1.08, 2.03), (0.05, 0.045, 0.06), WHITE, "eyeL_spark")
#   winking eye: white oval + a black curved closed-lid dash
add_ellipsoid((0.33, -0.86, 1.96), (0.235, 0.135, 0.305), WHITE, "eyeR_white")
add_ellipsoid((0.33, -1.00, 1.97), (0.175, 0.075, 0.05), BLACK, "eyeR_wink",
              rot=(0, 0, math.radians(-13)))

# === BEAK: wide, bright-orange, OPEN bill with a dark-red mouth interior.
add_ellipsoid((0.0, -0.95, 1.70), (0.46, 0.33, 0.165), ORANGE, "beak_upper",
              rot=(math.radians(15), 0, 0))
add_ellipsoid((0.0, -0.92, 1.51), (0.44, 0.31, 0.145), ORANGE, "beak_lower",
              rot=(math.radians(-15), 0, 0))
add_ellipsoid((0.0, -0.99, 1.605), (0.34, 0.18, 0.10), DARKRED, "mouth")

# === RED SCARF: a FLATTER wrapped band (thin front-to-back, tall), not a tube.
bpy.ops.mesh.primitive_torus_add(location=(0.0, 0.0, 1.45),
                                  major_radius=1.02, minor_radius=0.15,
                                  major_segments=72, minor_segments=24)
scarf = bpy.context.object
scarf.scale = (1.05, 1.0, 1.85)            # thin radially, tall in z -> flat band
scarf.rotation_euler = (math.radians(3), 0, 0)
scarf.name = "scarf"
scarf.data.materials.append(RED)
bpy.ops.object.shade_smooth()
# front knot where the scarf ties (pushed forward so it sits proud of the belly)
add_ellipsoid((-0.12, -1.14, 1.30), (0.27, 0.24, 0.28), RED, "scarf_knot")
# LONG forked hanging tail dropping down the FRONT of the belly (must sit in
# FRONT of the white belly, whose surface bulges to ~y=-1.30, or it is hidden).
add_box((-0.10, -1.35, 0.80), (0.185, 0.06, 0.50), RED, "scarf_tail",
        rot=(math.radians(-5), 0, math.radians(3)))
# swallowtail end: two angled tips forming a notch at the bottom
add_box((-0.20, -1.33, 0.36), (0.105, 0.06, 0.18), RED, "scarf_tip_a",
        rot=(0, 0, math.radians(17)))
add_box((0.02, -1.33, 0.36), (0.105, 0.06, 0.18), RED, "scarf_tip_b",
        rot=(0, 0, math.radians(-17)))

# === WINGS / flippers: black flattened ellipsoids hugging the sides.
add_ellipsoid((-1.10, 0.08, 0.88), (0.15, 0.44, 0.72), BLACK, "wingL",
              rot=(0, math.radians(17), math.radians(6)))
add_ellipsoid((1.10, 0.08, 0.88), (0.15, 0.44, 0.72), BLACK, "wingR",
              rot=(0, math.radians(-17), math.radians(-6)))

# === FEET: broad, flat, forward-splayed orange webbed feet at the base.
add_ellipsoid((-0.46, -0.55, -0.20), (0.44, 0.70, 0.15), ORANGE, "footL",
              rot=(0, 0, math.radians(15)))
add_ellipsoid((0.46, -0.55, -0.20), (0.44, 0.70, 0.15), ORANGE, "footR",
              rot=(0, 0, math.radians(-15)))

for o in bpy.data.objects:
    if o.type == "MESH":
        o.select_set(True)

print("BUILD_OK", len([o for o in bpy.data.objects if o.type == "MESH"]), "meshes")
