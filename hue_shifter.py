"""Main class file: hue_shifter.py"""

import random
from krita import ManagedColor, Extension, Krita
from PyQt5 import QtCore, QtWidgets

import numpy as np

# from .ok_gray_filter import OkGrayFilter

from .color_conversions import (
    okhsl_to_srgb,
    srgb_to_okhsl,
)

EXTENSION_ID = "pykrita_hue_shifter"
MENU_ENTRY = "Hue Shifter"


def find_canvas_widget():
    """ helper: find the canvas widget. Called once when toggling the extension."""
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


class CanvasReleaseFilter(QtCore.QObject):
    """ event filter """
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback

    def eventFilter(self, _obj, event):
        """ filter for mouse release events """
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            self.callback()
        return False


class HueShifter(Extension):
    """Main class for the Hue Shifter extension."""
    last_lightness = 0.0
    def __init__(self, parent):
        super().__init__(parent)
        self.krita = Krita.instance()
        self._filter = CanvasReleaseFilter(parent, self.on_mouse_release)
        self._canvas = None

    def setup(self):
        """ Run once at startup """
        print("\nHueShifter ready!\n")

    def createActions(self, window):
        """ Create the menu entries """
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
        """ Toggle the extension on/off """
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
    def okhsl_shift(cls, rgb: np.ndarray):
        """Using OK HSL to rotate the hue randomly."""
        h, s, l = srgb_to_okhsl(rgb)
        # Randomly rotate hue 0..1 and saturation: (+-) 0.15..0.85
        h = (h + random.random()) % 1.0
        s = (s + random.random()) % 1.0
        if s < 0.15:
            s += random.random() * 0.15
        if s > 0.85:
            s -= random.random() * 0.15
        # compensate lightness changes due to compute imperfections:
        if np.abs(cls.last_lightness - l) < 0.01:
            l = cls.last_lightness
        cls.last_lightness = l

        # print(f"New OK-HSL: {h:<.2f}, {s:<.2f}, {l:.6f}")
        # convert back to sRGB
        return okhsl_to_srgb(h, s, l)

    @staticmethod
    def check_in_gamut(r, g, b):
        """takes linear RGB values and checks if in gamut"""
        if not 0.0 <= r <= 1.0 or not 0.0 <= g <= 1.0 or not 0.0 <= b <= 1.0:
            raise ValueError(f"RGB {r, g, b} values out of gamut!")

    def on_mouse_release(self):
        """ callback on mouse release event """
        if not self._canvas:
            return
        view = self.krita.activeWindow().activeView()
        if not view:
            return

        components = view.foregroundColor().componentsOrdered()  # RGBA in [0.0..1.0]
        rgb = np.array(components[:3])
        alpha = components[3]
        # make ok hsl shift:
        r, g, b = self.okhsl_shift(rgb)
        # Set as new foreground color
        new_color = ManagedColor("RGBA", "U8", "")
        # Warning: uses BGR order because of Qt because of old graphics API, because of performance!
        new_color.setComponents([b, g, r, alpha])
        view.setForeGroundColor(new_color)
