import bpy
import math
import mathutils

# =============================================================================
# Tencent Hunyuan orb (腾讯混元 logo) -> 3D sphere carrying the three-armed swirl,
# wrapped so a face-on pinwheel "eye" reads from ALL FOUR turntable views
# (front/right/back/left -> camera dirs -Y,+X,+Y,-X).
#
# Approach: a matte flat-emission dual-vortex shader with per-render-pixel color
# derived from object-space position, so blade edges stay smooth at any render
# resolution. A face-on swirl "eye" reads from every turntable camera via the
# Y/X dual-vortex selected by nearer pole.
#
# The three arms are balanced so the CYAN arm reads as a genuinely distinct, fat
# third comma rather than a thin band squeezed between pale and deep:
#  (1) ARM WIDTHS: W_LB 0.42, W_CY 0.36 (deep ground = remainder ~0.22). Pale
#      stays the largest blade, cyan grows into a clear blade, and the deep
#      ground is kept small so it does not dominate side/back views.
#  (2) TAPER FLOOR 0.585 keeps the bright arms with fat "heads" toward the
#      shared center instead of pinching to thin tails -> the arms read as
#      teardrop/comma blades with bulb + hook, not constant-width bands.
#  Other key parameters: SWIRL 3.32 (tight log-spiral comma hook), thin white
#  separator rail SEP 0.012, tiny crisp 1.05deg glints (eps 0.0018), matte
#  material (roughness 1.0, specular 0, emission 0.55), dual-vortex Y/X selection
#  by nearer pole, sRGB->linear authored colors + Standard view transform +
#  exposure -0.5 to defeat factory AgX desaturation.
# =============================================================================

TWO_PI = 2.0 * math.pi
RADIUS = 1.0
SWIRL = 3.32                      # tight log-spiral wrap -> arms hook like commas
W_LB = 0.42                       # large pale light-blue blade (still dominant)
W_CY = 0.36                       # bright cyan blade ; deep ground = remainder ~0.22
SEP = 0.012                       # thin white separator rail before the pale blade
EPS = 0.006                       # soft-threshold half-width (anti-aliased edge)
TAPER_FLOOR = 0.585               # bright arms keep fat comma heads toward center


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(rgb):
    return (srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]),
            srgb_to_linear(rgb[2]), 1.0)


COL_DEEP = lin((0.027, 0.235, 0.92))    # deep royal blue ground / third blade
COL_LB = lin((0.62, 0.82, 0.965))       # pale light blue
COL_CYAN = lin((0.06, 0.66, 0.99))      # bright cyan blade
COL_WHITE = lin((1.0, 1.0, 1.0))


# one tiny glint near each of the four eyes (+/-Y, +/-X), offset into the curl
GLINT_DIRS = [
    mathutils.Vector((0.26, -0.93, 0.26)).normalized(),
    mathutils.Vector((0.26, 0.93, 0.26)).normalized(),
    mathutils.Vector((0.93, 0.26, 0.26)).normalized(),
    mathutils.Vector((-0.93, 0.26, 0.26)).normalized(),
]
GLINT_COS = math.cos(math.radians(1.05))

# -----------------------------------------------------------------------------
# build the sphere (round silhouette only; color comes from the shader)
# -----------------------------------------------------------------------------
bpy.ops.mesh.primitive_uv_sphere_add(segments=192, ring_count=96,
                                     radius=RADIUS, location=(0.0, 0.0, 0.0))
orb = bpy.context.active_object
orb.name = "HunyuanOrb"
for p in orb.data.polygons:
    p.use_smooth = True

# -----------------------------------------------------------------------------
# procedural swirl material
# -----------------------------------------------------------------------------
mat = bpy.data.materials.new("HunyuanSwirl")
mat.use_nodes = True
nt = mat.node_tree
nodes = nt.nodes
links = nt.links
nodes.clear()

_x = [0]


def _place(node):
    node.location = (_x[0], 0)
    _x[0] += 220
    return node


def _set_or_link(sock, val):
    if hasattr(val, "node"):
        links.new(val, sock)
    else:
        sock.default_value = val


def M(op, *ins, clamp=False):
    n = _place(nodes.new("ShaderNodeMath"))
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate(ins):
        _set_or_link(n.inputs[i], v)
    return n.outputs[0]


def maprange(val, fmin, fmax, tmin, tmax):
    n = _place(nodes.new("ShaderNodeMapRange"))
    n.interpolation_type = "SMOOTHSTEP"
    n.clamp = True
    _set_or_link(n.inputs[0], val)
    _set_or_link(n.inputs[1], fmin)
    _set_or_link(n.inputs[2], fmax)
    _set_or_link(n.inputs[3], tmin)
    _set_or_link(n.inputs[4], tmax)
    return n.outputs[0]


def soft_less(val, thr, eps=EPS):
    lo = M("SUBTRACT", thr, eps) if hasattr(thr, "node") else thr - eps
    hi = M("ADD", thr, eps) if hasattr(thr, "node") else thr + eps
    return maprange(val, lo, hi, 1.0, 0.0)


def soft_band(val, lo_thr, hi_thr, eps=EPS):
    # ~1 when lo_thr < val < hi_thr (used for the thin white rail)
    return M("MULTIPLY", soft_greater(val, lo_thr, eps), soft_less(val, hi_thr, eps))


def soft_greater(val, thr, eps=EPS):
    lo = M("SUBTRACT", thr, eps) if hasattr(thr, "node") else thr - eps
    hi = M("ADD", thr, eps) if hasattr(thr, "node") else thr + eps
    return maprange(val, lo, hi, 0.0, 1.0)


def mix_col(fac, a_col, b_col):
    n = _place(nodes.new("ShaderNodeMix"))
    n.data_type = "RGBA"
    _set_or_link(n.inputs[0], fac)
    _set_or_link(n.inputs[6], a_col)
    _set_or_link(n.inputs[7], b_col)
    return n.outputs[2]


def combine(vec):
    n = _place(nodes.new("ShaderNodeCombineXYZ"))
    n.inputs[0].default_value = vec[0]
    n.inputs[1].default_value = vec[1]
    n.inputs[2].default_value = vec[2]
    return n.outputs[0]


def dot(a_sock, vec):
    n = _place(nodes.new("ShaderNodeVectorMath"))
    n.operation = "DOT_PRODUCT"
    _set_or_link(n.inputs[0], a_sock)
    _set_or_link(n.inputs[1], combine(vec))
    return n.outputs[1]


# object-space position == unit direction d for a radius-1 sphere at origin
tc = _place(nodes.new("ShaderNodeTexCoord"))
pos = tc.outputs["Object"]
sepn = _place(nodes.new("ShaderNodeSeparateXYZ"))
links.new(pos, sepn.inputs[0])
X, Y, Z = sepn.outputs[0], sepn.outputs[1], sepn.outputs[2]


def clampm1p1(s):
    return M("MINIMUM", M("MAXIMUM", s, -1.0), 1.0)


def swirl(axis_dot, phi_a, phi_b):
    # returns (t in 0..1, taper s in 0..1 where s~1 at equator, ~0 at pole)
    polar = M("ARCCOSINE", clampm1p1(axis_dot))           # 0 at +pole .. pi at -pole
    phi = M("ARCTAN2", phi_a, phi_b)
    psi = M("SUBTRACT", phi, M("MULTIPLY", polar, SWIRL))
    u = M("MULTIPLY", psi, 1.0 / TWO_PI)
    t = M("WRAP", u, 1.0, 0.0)
    s = M("SINE", polar)                                  # 0 at poles, 1 at equator
    return t, s


def blade_color(t, s):
    # Each threshold scaled by taper: bright blades pinch toward the pole
    # (pointed comma tails into a tight shared center), deep ground grows there.
    taper = maprange(s, 0.0, 1.0, TAPER_FLOOR, 1.0)
    thr_lb = M("MULTIPLY", taper, W_LB)
    thr_cy = M("MULTIPLY", taper, W_LB + W_CY)
    thr_sep = M("MULTIPLY", taper, W_LB + SEP)            # white rail just past pale
    # deep ground -> cyan blade
    c = mix_col(soft_less(t, thr_cy), COL_DEEP, COL_CYAN)
    # thin white separator rail between cyan and pale-blue
    c = mix_col(soft_band(t, thr_lb, thr_sep), c, COL_WHITE)
    # large pale light-blue blade
    c = mix_col(soft_less(t, thr_lb), c, COL_LB)
    return c


# Y vortex: axis_dot=y, phi=atan2(x,z)
tY, sY = swirl(Y, X, Z)
colY = blade_color(tY, sY)
# X vortex: axis_dot=x, phi=atan2(-y,z)
tX, sX = swirl(X, M("MULTIPLY", Y, -1.0), Z)
colX = blade_color(tX, sX)

# select vortex by nearer pole: use Y when |y| >= |x|
ay = M("ABSOLUTE", Y)
ax = M("ABSOLUTE", X)
diff = M("SUBTRACT", ay, ax)
selY = maprange(diff, -0.04, 0.04, 0.0, 1.0)
col = mix_col(selY, colX, colY)

# overlay tiny white glints
for g in GLINT_DIRS:
    fac = soft_greater(dot(pos, (g.x, g.y, g.z)), GLINT_COS, eps=0.0018)
    col = mix_col(fac, col, COL_WHITE)

# principled output
bsdf = _place(nodes.new("ShaderNodeBsdfPrincipled"))
links.new(col, bsdf.inputs["Base Color"])
if "Emission Color" in bsdf.inputs:
    links.new(col, bsdf.inputs["Emission Color"])
if "Emission Strength" in bsdf.inputs:
    bsdf.inputs["Emission Strength"].default_value = 0.55
bsdf.inputs["Roughness"].default_value = 1.0
bsdf.inputs["Metallic"].default_value = 0.0
for sname in ("Specular IOR Level", "Specular"):
    if sname in bsdf.inputs:
        try:
            bsdf.inputs[sname].default_value = 0.0
        except Exception:
            pass

out = _place(nodes.new("ShaderNodeOutputMaterial"))
links.new(bsdf.outputs[0], out.inputs["Surface"])
orb.data.materials.append(mat)

# color management: defeat factory AgX desaturation
try:
    vs_ = bpy.context.scene.view_settings
    vs_.view_transform = "Standard"
    try:
        vs_.look = "None"
    except Exception:
        pass
    vs_.exposure = -0.5
    vs_.gamma = 1.0
except Exception:
    pass

bpy.context.view_layer.update()
