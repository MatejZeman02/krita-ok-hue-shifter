"""
Converted from TypeScript to Python. Working in normalized range only.

Original source: https://github.com/holbrookdev/ok-color-picker/blob/main/

SOURCE: https://github.com/bottosson/bottosson.github.io/blob/master/misc/colorpicker/colorconversion.js
LICENSE: https://github.com/bottosson/bottosson.github.io/blob/master/misc/colorpicker/License.txt
Copyright (c) 2021 Björn Ottosson
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List
import re

import numpy as np


def rgb_to_hsl(r: float, g: float, b: float):

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    h = 0
    s = 0
    l = (max_val + min_val) / 2

    if max_val == min_val:
        h = s = 0
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        elif max_val == b:
            h = (r - g) / d + 4
        h /= 6

    return [h, s, l]


def hsl_to_rgb(h: float, s: float, l: float):
    if s == 0:
        r = g = b = l
    else:

        def hue_to_rgb(p: float, q: float, t: float):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

    return [r, g, b]


def rgb_to_hsv(r: float, g: float, b: float):
    """Convert sRGB (non-linear) to HSV"""

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    h = 0
    s = 0
    v = max_val

    d = max_val - min_val
    s = 0 if max_val == 0 else d / max_val

    if max_val == min_val:
        h = 0  # achromatic
    else:
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        elif max_val == b:
            h = (r - g) / d + 4
        h /= 6

    return [h, s, v]


def hsv_to_rgb(h: float, s: float, v: float):
    """Convert HSV to sRGB (non-linear)"""
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    # elif i % 6 == 5:
    else:
        r, g, b = v, p, q

    return [r, g, b]


def channel_rgb_to_srgb(a: float):
    """converts normalized linear RGB value to non-linear sRGB value"""
    a = np.asarray(a)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)


def srgb_transfer_function(a: float | np.ndarray):
    """converts normalized linear RGB value to non-linear sRGB value"""
    if isinstance(a, (list, np.ndarray, tuple)):
        return np.array([channel_rgb_to_srgb(c) for c in a])
    a = channel_rgb_to_srgb(a)
    return np.array(a)


def channel_srgb_to_rgb(a: float):
    a = np.asarray(a)
    return np.where(a > 0.04045, np.power((a + 0.055) / 1.055, 2.4), a / 12.92)


def srgb_transfer_function_inv(a: float | np.ndarray):
    """converts normalized non-linear sRGB value to linear RGB value"""
    if isinstance(a, (list, np.ndarray, tuple)):
        return np.array([channel_srgb_to_rgb(c) for c in a])
    a = channel_srgb_to_rgb(a)
    return np.array(a)


def color_to_array(args, x=0, y=0, z=0) -> np.ndarray:
    """Convert various input formats to a numpy array of three floats or one array."""
    if len(args) == 1:
        if isinstance(args[0], (list, tuple, np.ndarray)):
            return np.array(*args)
        else:  # int/float
            return np.repeat(args[0], 3)
    elif len(args) == 3:
        return np.array(args)
    return np.array([x, y, z])


def linear_srgb_to_oklab(*args, r=0, g=0, b=0) -> List[float]:
    """Takes normalized linear sRGB values in range 0-1 and returns OKLab values"""
    rgb = color_to_array(args, r, g, b)
    if all(x == 0 for x in rgb):
        return [0, 0, 0]

    # Linear sRGB to LMS matrix
    M = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = M @ rgb
    lms_ = np.cbrt(lms)

    # LMS to OKLab matrix
    M2 = np.array(
        [
            [0.2104542553, 0.793617785, -0.0040720468],
            [1.9779984951, -2.428592205, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.808675766],
        ]
    )
    oklab = M2 @ lms_

    return oklab.tolist()


def srgb_to_oklab(*args, r=0, g=0, b=0):
    srgb = color_to_array(args, r, g, b)
    lab = linear_srgb_to_oklab(srgb_transfer_function_inv(srgb))
    return lab


def oklab_to_linear_srgb(*args, L=0, a=0, b=0) -> List[float]:
    lab = color_to_array(args, L, a, b)
    if all(x == 0 for x in lab):
        return [0, 0, 0]

    # Transformation matrix for l_, m_, s_
    LMS_M = np.array(
        [
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.291485548],
        ]
    )
    lms_ = LMS_M @ lab
    lms = lms_**3

    # Transformation matrix for linear sRGB
    RGB_M = np.array(
        [
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.707614701],
        ]
    )
    rgb = RGB_M @ lms

    return rgb.tolist()


def oklab_to_srgb(*args, l=0, a=0, b=0):
    lab = color_to_array(args, l, a, b)
    srgb = srgb_transfer_function(oklab_to_linear_srgb(lab))
    return srgb


def toe(x: float) -> float:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1 + k_1) / (1 + k_2)
    return 0.5 * (
        k_3 * x - k_1 + np.sqrt((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x)
    )


def toe_inv(x: float) -> float:
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1 + k_1) / (1 + k_2)
    return (x**2 + k_1 * x) / (k_3 * (x + k_2))


# Finds the maximum saturation possible for a given hue that fits in sRGB
# Saturation here is defined as S = C/L
# a and b must be normalized so a^2 + b^2 == 1
def compute_max_saturation(a: float, b: float) -> float:
    # Max saturation will be when one of r, g or b goes below zero.

    # Select different coefficients depending on which component goes below zero first
    if -1.88170328 * a - 0.80936493 * b > 1:
        # Red component
        k0, k1, k2, k3, k4 = 1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245
        wl, wm, ws = 4.0767416621, -3.3077115913, 0.2309699292
    elif 1.81444104 * a - 1.19445276 * b > 1:
        # Green component
        k0, k1, k2, k3, k4 = 0.73956515, -0.45954404, 0.08285427, 0.1254107, 0.14503204
        wl, wm, ws = -1.2684380046, 2.6097574011, -0.3413193965
    else:
        # Blue component
        k0, k1, k2, k3, k4 = (
            1.35733652,
            -0.00915799,
            -1.1513021,
            -0.50559606,
            0.00692167,
        )
        wl, wm, ws = -0.0041960863, -0.7034186147, 1.707614701

    # Approximate max saturation using a polynomial:
    S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

    # Do one step Halley's method to get closer
    k_l = 0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.291485548 * b

    l_ = 1 + S * k_l
    m_ = 1 + S * k_m
    s_ = 1 + S * k_s

    l = l_**3
    m = m_**3
    s = s_**3

    l_dS = 3 * k_l * l_**2
    m_dS = 3 * k_m * m_**2
    s_dS = 3 * k_s * s_**2

    l_dS2 = 6 * k_l**2 * l_
    m_dS2 = 6 * k_m**2 * m_
    s_dS2 = 6 * k_s**2 * s_

    f = wl * l + wm * m + ws * s
    f1 = wl * l_dS + wm * m_dS + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

    if f1 == 0:
        f1 = 1e-10  # avoid division by zero
    if f1**2 - 0.5 * f * f2 == 0:
        f2 = 1e-10  # avoid division by zero

    S = S - (f * f1) / (f1**2 - 0.5 * f * f2)

    return S


def find_cusp(a: float, b: float) -> list:
    # First, find the maximum saturation (saturation S = C/L)
    S_cusp = compute_max_saturation(a, b)

    # Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
    rgb_at_max = oklab_to_linear_srgb(1, S_cusp * a, S_cusp * b)
    L_cusp = np.cbrt(1 / max(max(rgb_at_max[0], rgb_at_max[1]), rgb_at_max[2]))
    C_cusp = L_cusp * S_cusp

    return [L_cusp, C_cusp]


def find_gamut_intersection(a, b, L1, C1, L0, cusp=None):
    if cusp is None:
        # Find the cusp of the gamut triangle
        cusp = find_cusp(a, b)

    # Find the intersection for upper and lower half separately
    if (L1 - L0) * cusp[1] - (cusp[0] - L0) * C1 <= 0:
        # Lower half
        t = (cusp[1] * L0) / (C1 * cusp[0] + cusp[1] * (L0 - L1))
    else:
        # Upper half
        # First intersect with triangle
        t = (cusp[1] * (L0 - 1)) / (C1 * (cusp[0] - 1) + cusp[1] * (L0 - L1))

        # Then one step Halley's method
        dL = L1 - L0
        dC = C1

        k_l = 0.3963377774 * a + 0.2158037573 * b
        k_m = -0.1055613458 * a - 0.0638541728 * b
        k_s = -0.0894841775 * a - 1.291485548 * b

        l_dt = dL + dC * k_l
        m_dt = dL + dC * k_m
        s_dt = dL + dC * k_s

        # If higher accuracy is required, 2 or 3 iterations of the following block can be used:
        L = L0 * (1 - t) + t * L1
        C = t * C1

        l_ = L + C * k_l
        m_ = L + C * k_m
        s_ = L + C * k_s

        l = l_**3
        m = m_**3
        s = s_**3

        ldt = 3 * l_dt * l_**2
        mdt = 3 * m_dt * m_**2
        sdt = 3 * s_dt * s_**2

        ldt2 = 6 * l_dt**2 * l_
        mdt2 = 6 * m_dt**2 * m_
        sdt2 = 6 * s_dt**2 * s_

        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
        r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
        r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2

        u_r = r1 / (r1**2 - 0.5 * r * r2)
        t_r = -r * u_r

        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1
        g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
        g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2

        u_g = g1 / (g1**2 - 0.5 * g * g2)
        t_g = -g * u_g

        b = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s - 1
        b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.707614701 * sdt
        b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.707614701 * sdt2

        u_b = b1 / (b1**2 - 0.5 * b * b2)
        t_b = -b * u_b

        t_r = t_r if u_r >= 0 else 10e5
        t_g = t_g if u_g >= 0 else 10e5
        t_b = t_b if u_b >= 0 else 10e5

        t += min(t_r, t_g, t_b)

    return t


def get_ST_max(a_, b_, cusp=None):
    if not cusp:
        cusp = find_cusp(a_, b_)

    L = cusp[0]
    if L == 1:
        L = 1 - 1e-10  # avoid division by zero
    if L == 0:
        L = 1e-10  # avoid division by zero
    C = cusp[1]
    return [C / L, C / (1 - L)]


def get_Cs(L, a_, b_):
    cusp = find_cusp(a_, b_)

    C_max = find_gamut_intersection(a_, b_, L, 1, L, cusp)
    ST_max = get_ST_max(a_, b_, cusp)

    S_mid = 0.11516993 + 1 / (
        +7.4477897
        + 4.1590124 * b_
        + a_
        * (
            -2.19557347
            + 1.75198401 * b_
            + a_
            * (
                -2.13704948
                - 10.02301043 * b_
                + a_ * (-4.24894561 + 5.38770819 * b_ + 4.69891013 * a_)
            )
        )
    )

    T_mid = 0.11239642 + 1 / (
        +1.6132032
        - 0.68124379 * b_
        + a_
        * (
            +0.40370612
            + 0.90148123 * b_
            + a_
            * (
                -0.27087943
                + 0.6122399 * b_
                + a_ * (+0.00299215 - 0.45399568 * b_ - 0.14661872 * a_)
            )
        )
    )

    k = C_max / min(L * ST_max[0], (1 - L) * ST_max[1])

    C_a = L * S_mid
    C_b = (1 - L) * T_mid
    C_mid = (
        0.9
        * k
        * (1 / (1 / (C_a ** 4) + 1 / (C_b **4))) ** 0.25
    )

    C_a = L * 0.4
    C_b = (1 - L) * 0.8
    C_0 = (1 / (1 / (C_a **2) + 1 / (C_b **2))) ** 0.5

    return [C_0, C_mid, C_max]


def okhsl_to_srgb(h, s, l):
    if l == 1:
        return [1, 1, 1]
    elif l == 0:
        return [0, 0, 0]

    a_ = np.cos(2 * np.pi * h)
    b_ = np.sin(2 * np.pi * h)
    L = toe_inv(l)

    Cs = get_Cs(L, a_, b_)
    C_0 = Cs[0]
    C_mid = Cs[1]
    C_max = Cs[2]

    if s < 0.8:
        t = 1.25 * s
        k_0 = 0
        k_1 = 0.8 * C_0
        k_2 = 1 - k_1 / C_mid
    else:
        t = 5 * (s - 0.8)
        k_0 = C_mid
        k_1 = (0.2 * C_mid * C_mid * 1.25 * 1.25) / C_0
        k_2 = 1 - k_1 / (C_max - C_mid)

    C = k_0 + (t * k_1) / (1 - k_2 * t)

    rgb = oklab_to_linear_srgb(L, C * a_, C * b_)
    return srgb_transfer_function(rgb)


def srgb_to_okhsl(*args, r=0, g=0, b=0) -> List[float]:
    """Converts non linear sRGB (0-1) to OKHSL [0-1, 0-1, 0-1]"""
    rgb = color_to_array(args, r, g, b)
    if all(x == 0 for x in rgb):
        return [0, 0, 0]

    rgb_lin = srgb_transfer_function_inv(rgb)
    lab = linear_srgb_to_oklab(rgb_lin)

    C = np.sqrt(lab[1] * lab[1] + lab[2] * lab[2])
    a_ = lab[1] / C
    b_ = lab[2] / C

    L = lab[0]
    h = 0.5 + (0.5 * np.arctan2(-lab[2], -lab[1])) / np.pi

    Cs = get_Cs(L, a_, b_)
    C_0 = Cs[0]
    C_mid = Cs[1]
    C_max = Cs[2]

    if C < C_mid:
        k_0 = 0
        k_1 = 0.8 * C_0
        k_2 = 1 - k_1 / C_mid

        t = (C - k_0) / (k_1 + k_2 * (C - k_0))
        s = t * 0.8
    else:
        k_0 = C_mid
        k_1 = (0.2 * C_mid * C_mid * 1.25 * 1.25) / C_0
        k_2 = 1 - k_1 / (C_max - C_mid)

        t = (C - k_0) / (k_1 + k_2 * (C - k_0))
        s = 0.8 + 0.2 * t

    l = toe(L)
    return [h, s, l]


def okhsv_to_srgb(h, s, v):
    a_ = np.cos(2 * np.pi * h)
    b_ = np.sin(2 * np.pi * h)

    ST_max = get_ST_max(a_, b_)
    S_max = ST_max[0]
    S_0 = 0.5
    T = ST_max[1]
    k = 1 - S_0 / S_max

    L_v = 1 - (s * S_0) / (S_0 + T - T * k * s)
    C_v = (s * T * S_0) / (S_0 + T - T * k * s)

    L = v * L_v
    C = v * C_v

    L_vt = toe_inv(L_v)
    C_vt = (C_v * L_vt) / L_v

    L_new = toe_inv(L)
    C = (C * L_new) / L
    L = L_new

    rgb_scale = oklab_to_linear_srgb(L_vt, a_ * C_vt, b_ * C_vt)
    scale_L = np.cbrt(1 / max(rgb_scale[0], rgb_scale[1], rgb_scale[2], 0))

    L = L * scale_L
    C = C * scale_L

    rgb = oklab_to_linear_srgb(L, C * a_, C * b_)
    srgb = srgb_transfer_function(rgb)
    return srgb.tolist()


def srgb_to_okhsv(*args, r=0, g=0, b=0) -> List[float]:
    """Converts non linear sRGB (0-1) to OKHSV [0-1, 0-1, 0-1]"""
    rgb = color_to_array(args, r, g, b)
    if all(x == 0 for x in rgb):
        return [0, 0, 0]

    rgb_lin = srgb_transfer_function_inv(rgb)
    lab = linear_srgb_to_oklab(rgb_lin)

    C = (lab[1] ** 2 + lab[2] ** 2) ** 0.5
    if C == 0:
        C = 1e-10  # avoid division by zero
    a_ = lab[1] / C
    b_ = lab[2] / C

    L = lab[0]
    h = 0.5 + (0.5 * np.arctan2(-lab[2], -lab[1])) / np.pi

    ST_max = get_ST_max(a_, b_)
    S_max = ST_max[0]
    S_0 = 0.5
    T = ST_max[1]
    k = 1 - S_0 / S_max

    t = T / (C + L * T)
    L_v = t * L
    C_v = t * C

    L_vt = toe_inv(L_v)
    if L_v == 0:
        L_v = 1e-10  # avoid division by zero
    C_vt = (C_v * L_vt) / L_v

    rgb_scale = oklab_to_linear_srgb(L_vt, a_ * C_vt, b_ * C_vt)
    scale_L = (1 / max(rgb_scale[0], rgb_scale[1], rgb_scale[2], 0)) ** (1 / 3)

    L /= scale_L
    C /= scale_L

    C = (C * toe(L)) / L
    L = toe(L)

    v = L / L_v
    s = ((S_0 + T) * C_v) / (T * S_0 + T * k * C_v)

    return [h % 1, s % 1, v % 1]


def hex_to_rgb(hex):
    if hex[0] == "#":
        hex = hex[1:]

    if re.match(r"^([0-9a-f]{3})$", hex, re.I):
        r = (int(hex[0], 16) / 15) * 255
        g = (int(hex[1], 16) / 15) * 255
        b = (int(hex[2], 16) / 15) * 255
        return [r, g, b]
    elif re.match(r"^([0-9a-f]{6})$", hex, re.I):
        r = int(hex[:2], 16)
        g = int(hex[2:4], 16)
        b = int(hex[4:6], 16)
        return [r, g, b]
    elif re.match(r"^([0-9a-f]{1})$", hex, re.I):
        a = (int(hex[0], 16) / 15) * 255
        return [a, a, a]
    elif re.match(r"^([0-9a-f]{2})$", hex, re.I):
        a = int(hex, 16)
        return [a, a, a]
    else:
        return None


def rgb_to_hex(r, g, b):
    def component_to_hex(x):
        hex_str = hex(round(x))[2:]
        return "0" + hex_str if len(hex_str) == 1 else hex_str

    return "#" + component_to_hex(r) + component_to_hex(g) + component_to_hex(b)
