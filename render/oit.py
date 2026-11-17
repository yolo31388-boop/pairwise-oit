"""Order-independent transparency (OIT) helpers.

Data layouts
------------
``resolve`` fragments are ``(r, g, b, depth, alpha)`` in sRGB colour space.
``build_lists`` fragments are ``(depth, alpha)``; a per-pixel linked list only
needs to know where the fragment sits and how opaque it is.

A weighted-blended OIT resolve works in linear light: unlit (sRGB) colours are
decoded, accumulated in linear space, then re-encoded for display.  Compositing
in sRGB directly would produce the washed-out grey of the original code.
"""

from __future__ import annotations

_EPS = 1e-12


def _clamp01(value):
    """Clamp *value* to the alpha range [0, 1]."""
    value = float(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def oit_weight(alpha: float, depth: float, brightness: float = 1.0) -> float:
    """Weight for one fragment in weighted blended OIT.

    The old implementation returned the bare alpha, so the blend result
    depended on draw order.  A usable weight has to fall off with depth (near
    fragments matter most) and be normalised against the fragment's luminance
    so HDR-bright surfaces cannot dominate purely because they are bright.

        w = alpha / ((depth + 1) * max(brightness, 1))

    Depth is measured along the view ray; ``depth + 1`` keeps the weight
    finite at the near plane.  Brightness is clamped at 1 so ordinary LDR
    colours are not boosted — only genuinely over-bright (HDR) fragments are
    normalised down.
    """
    alpha = _clamp01(alpha)
    depth = max(0.0, float(depth))
    luminance = max(1.0, float(brightness))
    return alpha / ((depth + 1.0) * luminance)


def _fragment_depth(fragment):
    """Depth of a ``(depth, ...)`` fragment."""
    return float(fragment[0])


def build_lists(fragments, capacity: int):
    """Build a bounded per-pixel fragment pool.

    When the pool is full the farthest fragment is evicted instead of silently
    dropping the incoming one.  Dropping arrivals produced holes at the front
    of dense smoke: with a fixed budget the near fragments are exactly the
    ones that should be kept.

    Returns the surviving fragments (at most ``capacity`` of them).  Fragments
    are kept in the order they entered the pool; evictions overwrite the slot
    of the farthest survivor.
    """
    capacity = int(capacity)
    if capacity <= 0:
        return []

    pool = []
    for fragment in fragments:
        depth = _fragment_depth(fragment)
        if len(pool) < capacity:
            pool.append(fragment)
            continue

        # Replace the farthest resident when the new fragment is nearer.
        farthest = max(range(len(pool)), key=lambda i: _fragment_depth(pool[i]))
        if depth < _fragment_depth(pool[farthest]):
            pool[farthest] = fragment
        # Otherwise the incoming fragment is the farthest and is discarded.

    return pool


def sort_fragments(fragments):
    """Sort ``(id, depth)`` fragments front-to-back by depth.

    Python's ``sorted`` is stable, so fragments at equal depth keep their
    original relative order.  (Object-centre sorting is what made long lasers
    intersect smoke incorrectly; callers must pass the per-pixel/sample depth
    they actually want sorted.)
    """
    return sorted(fragments, key=lambda fragment: fragment[1])


def resolve(fragments, background, gamma: float = 2.2):
    """Resolve ``(r, g, b, depth, alpha)`` fragments with weighted blending.

    Accumulation and revealage are evaluated in linear space:

    * ``accum``  = sum(colour * weight)
    * ``denom``  = sum(weight)
    * ``reveal`` = product(1 - alpha)

    The blended colour is
    ``accum / denom * (1 - reveal) + background * reveal``,
    then encoded back to sRGB when ``gamma`` is not 1.

    Because the accumulation is a sum and the revealage a product, the result
    is order independent; fragments are nevertheless sorted by depth first so
    floating-point summation is deterministic.
    """
    gamma = float(gamma)
    linear_encode = gamma != 1.0 and gamma > 0.0

    background = tuple(float(channel) for channel in background)
    if linear_encode:
        background_linear = tuple(channel ** gamma for channel in background)
    else:
        background_linear = background

    ordered = sorted(fragments, key=lambda fragment: fragment[3])

    accum = [0.0, 0.0, 0.0]
    accum_weight = 0.0
    reveal = 1.0

    for fragment in ordered:
        red, green, blue, depth, alpha = fragment[:5]
        alpha = _clamp01(alpha)

        if linear_encode:
            red, green, blue = red ** gamma, green ** gamma, blue ** gamma

        # Brightness of the linear colour drives the luminance normalisation.
        brightness = max(red, green, blue)
        weight = oit_weight(alpha, depth, brightness)

        accum[0] += red * weight
        accum[1] += green * weight
        accum[2] += blue * weight
        accum_weight += weight
        reveal *= 1.0 - alpha

    if accum_weight > _EPS:
        coverage = 1.0 - reveal
        result = [
            accum[i] / accum_weight * coverage + background_linear[i] * reveal
            for i in range(3)
        ]
    else:
        # No contributing fragments: the background shows through.
        result = list(background_linear)

    if linear_encode:
        result = [channel ** (1.0 / gamma) for channel in result]

    return tuple(result)


def msaa_heads(width: int, height: int, samples: int):
    """Allocate one fragment-list head per pixel *per sample*.

    The previous code shared a single head between all samples of a pixel,
    which let one sample's fragment chain leak into its neighbours and
    produced striping on MSAA edges.  Heads are returned as a flat list,
    indexed ``(y * width + x) * samples + s``; ``-1`` means the chain is empty.
    """
    width = max(0, int(width))
    height = max(0, int(height))
    samples = max(0, int(samples))
    return [-1] * (width * height * samples)
