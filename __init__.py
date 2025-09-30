"""Init of classes"""
from .hue_shifter import HueShifter

# And add the extension to Krita's list of extensions:
app = Krita.instance()
# Instantiate class:
extension = HueShifter(parent = app)
app.addExtension(extension)
