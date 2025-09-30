"""Not working filter that converts to OK gray. Use soft proofing instead."""

import numpy as np

from krita import Filter

from .color_conversions import (
    linear_srgb_to_oklab,
    oklab_to_linear_srgb,
    srgb_transfer_function_inv,
    srgb_transfer_function,
)


def color_to_ok_gray(color_srgb):
    lab = linear_srgb_to_oklab(srgb_transfer_function_inv(color_srgb))
    lab[1] = 0
    lab[2] = 0
    lin_rgb = oklab_to_linear_srgb(lab)
    srgb = srgb_transfer_function(lin_rgb)
    srgb = np.clip(srgb, 0, 1)
    return srgb


class OkGrayFilter(Filter):
    def __init__(self):
        super().__init__()
        self.setName("OK Gray Filter")

    def apply(self, node, x, y, w, h):
        """Apply filter to rectangle (x,y,w,h) of node."""
        # get pixel data from node
        # pixel_data = node.pixelData(x, y, w, h)
        # convert pixel_data via your color_to_ok_gray
        # set it back via node.setPixelData(...)
        print("filter apply() called. Not implemented!")
        return True
