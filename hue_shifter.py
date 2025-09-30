"""Main class file: hue_shifter.py"""

import random
from krita import ManagedColor, Extension, Krita
from PyQt5 import QtCore, QtWidgets

from scipy.interpolate import interp1d
import numpy as np


# from .ok_gray_filter import OkGrayFilter

from .color_conversions import (
    okhsl_to_srgb,
    srgb_to_okhsl,
)

EXTENSION_ID = "pykrita_hue_shifter"
MENU_ENTRY = "Hue Shifter"

# semaphore initial value: shifts per x mouse releases
SEMAPHORE_INIT = 2


def find_canvas_widget():
    """helper: find the canvas widget. Called once when toggling the extension."""
    app = Krita.instance()
    win = app.activeWindow()
    if win is None:
        return None
    qwin = win.qwindow()
    if qwin is None:
        return None
    central = qwin.centralWidget()
    if central is None:
        return None
    return central.findChild(QtWidgets.QOpenGLWidget)

def map_lightness(L):
    """map L shrunken to [0.3,0.7] range"""
    L = np.clip(L, 0.0, 1.0)
    if L <= 0.3:
        return 0.0
    elif L >= 0.7:
        return 1.0
    return (L - 0.3) / (0.7 - 0.3) # normalize to [0,1]


def prepare_distribution(hues, probs):
    """prepare a hue distribution from 'okhsl_h' and 'probability' columns"""
    # sort by hue
    idx = np.argsort(hues)
    hues = hues[idx]
    probs = probs[idx]

    # add wrap-around for smooth 0=360 continuity
    hues = np.append(hues, hues[0] + 360.0)
    probs = np.append(probs, probs[0])

    # linear interpolation
    xs = np.linspace(0, 360, 1000)
    ys = np.interp(xs, hues, probs)

    # normalize area
    area = np.trapz(ys, xs)
    ys /= area

    return xs, ys, area


class CanvasReleaseFilter(QtCore.QObject):
    """event filter"""

    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback

    def eventFilter(self, _obj, event):
        """filter for mouse release events"""
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            self.callback()
        return False


class HueShifter(Extension):
    """Main class for the Hue Shifter extension."""

    # class variables:
    last_lightness = 0.0
    last_rgb = None
    shifter_semaphore = SEMAPHORE_INIT + random.randint(0, 2)

    # Defining the light and dark distributions:
    hues_light = np.array(
        [0, 28, 65, 106, 120, 150, 190, 210, 265, 327, 360], dtype=float
    )
    probs_light = np.array(
        [0.05, 0.475, 0.6, 0.75, 0.45, 0.25, 0.01, 0.01, 0.15, 0.0, 0.05], dtype=float
    )

    hues_dark = np.array(
        [0, 28, 65, 106, 120, 150, 190, 210, 265, 327, 360], dtype=float
    )
    probs_dark = np.array(
        [0.1, 0.48, 0.15, 0.0, 0.12, 0.24, 0.15, 0.15, 0.75, 0.15, 0.1], dtype=float
    )

    # Prepare distributions
    hues_light, prob_light, _ = prepare_distribution(hues_light, probs_light)
    hues_dark, prob_dark, _ = prepare_distribution(hues_dark, probs_dark)

    # Normalize probabilities
    prob_light /= prob_light.sum()
    prob_dark /= prob_dark.sum()

    def __init__(self, parent):
        super().__init__(parent)
        self.krita = Krita.instance()
        self._filter = CanvasReleaseFilter(parent, self.on_mouse_release)
        self._canvas = None

    def setup(self):
        """Run once at startup"""
        print("\nHueShifter ready!\n")

    def createActions(self, window):
        """Create the menu entries"""
        action = window.createAction(EXTENSION_ID, MENU_ENTRY, "tools")
        action.setCheckable(True)
        action.triggered.connect(self.toggle)

        # action = window.createAction(
        #     "ok_gray_filter_node", "Apply OK-Gray Filter", "tools"
        # )
        # action.triggered.connect(self.apply_ok_gray_filter)

    # def apply_ok_gray_filter(self):
    #     doc = Krita.instance().activeDocument()
    #     node = doc.activeNode()
    #     if node is None:
    #         return
    #     # Create a filter layer over the current node
    #     f = OkGrayFilter()
    #     # Optionally wrap in a FilterLayer, or apply directly
    #     # You can also create a new paint layer and copy output
    #     print("doing nothing right now. To be implemented...")
    #     f.apply(node, 0, 0, node.width(), node.height())
    #     doc.refreshProjection()

    def toggle(self, checked):
        """Toggle the extension on/off"""
        canvas = find_canvas_widget()
        if not canvas:
            print("Canvas not found")
            return
        if checked:
            canvas.installEventFilter(self._filter)
            self._canvas = canvas
            print("Hue shifter enabled")
        else:
            if self._canvas:
                # self._canvas.removeEventFilter(self._filter)
                self._canvas = None
            print("Hue shifter disabled")

    @classmethod
    def get_hue_sampler(cls, l, resolution=36000):
        """
        L: Lightness in [0,1]
        Returns a function to sample hue from a continuous probability distribution.
        """
        # Hue grid
        hue_grid = np.linspace(0, 360, resolution, endpoint=False)

        # Interpolate probability functions on the grid
        interp_light = interp1d(
            cls.hues_light, cls.prob_light, kind="linear", fill_value="extrapolate"
        )
        interp_dark = interp1d(
            cls.hues_dark, cls.prob_dark, kind="linear", fill_value="extrapolate"
        )

        p_light_grid = interp_light(hue_grid)
        p_dark_grid = interp_dark(hue_grid)

        # Blend depending on L
        p_interp = l * p_light_grid + (1 - l) * p_dark_grid

        # Normalize to sum = 1
        p_interp = np.maximum(p_interp, 0)  # safety: avoid negatives
        p_interp /= p_interp.sum()

        # Build CDF
        cdf = np.cumsum(p_interp)

        def sample_hue():
            r = np.random.rand(1)
            idx = np.searchsorted(cdf, r)
            return hue_grid[idx][0]

        return sample_hue

    @classmethod
    def okhsl_shift(cls, rgb: np.ndarray):
        """Using OK HSL to rotate the hue randomly."""
        h, s, l = srgb_to_okhsl(rgb)
        hue_sampler = cls.get_hue_sampler(map_lightness(l))
        h = hue_sampler() / 360.0  # convert to [0,1]
        # h = random.random()
        # Randomly rotate saturation (+-) 0.25..0.85
        s = random.random()
        if s < 0.25:
            s += random.random() * l
        elif s > 0.85:
            s -= random.random() * l
        # compensate lightness changes due to compute imperfections:
        if np.abs(cls.last_lightness - l) < 0.01:
            l = cls.last_lightness
        cls.last_lightness = l

        # print(f"New OK-HSL: {h:<.2f}, {s:<.2f}, {l:.6f}")
        if h < 0.0 or h > 1.0:
            raise ValueError(f"Hue {h} out of range!")
        # convert back to sRGB
        return okhsl_to_srgb(h, s, l)

    @staticmethod
    def check_in_gamut(r, g, b):
        """takes linear RGB values and checks if in gamut"""
        if not 0.0 <= r <= 1.0 or not 0.0 <= g <= 1.0 or not 0.0 <= b <= 1.0:
            raise ValueError(f"RGB {r, g, b} values out of gamut!")

    def on_mouse_release(self):
        """callback on mouse release event"""
        if not self._canvas:
            return
        view = self.krita.activeWindow().activeView()
        if not view:
            return
        # print(self.shifter_semaphore)

        components = view.foregroundColor().componentsOrdered()  # RGBA in [0.0..1.0]
        rgb = np.array(components[:3])
        if self.last_rgb is None:
            self.last_rgb = rgb
        alpha = components[3]

        if not np.allclose(rgb, self.last_rgb, rtol=1e-2, atol=1e-2):
            self.last_rgb = rgb
            self.shifter_semaphore = SEMAPHORE_INIT + random.randint(0, 2)
            return
        elif self.shifter_semaphore > 1:  # if colors were equal
            self.shifter_semaphore -= 1
            return
        self.shifter_semaphore = SEMAPHORE_INIT + random.randint(0, 2)

        # make ok hsl shift:
        r, g, b = self.okhsl_shift(rgb)
        # Set as new foreground color
        new_color = ManagedColor("RGBA", "U8", "")
        # Warning: uses BGR order because of Qt because of old graphics API, because of performance!
        new_color.setComponents([b, g, r, alpha])
        view.setForeGroundColor(new_color)
        self.last_rgb = np.array([r, g, b])
