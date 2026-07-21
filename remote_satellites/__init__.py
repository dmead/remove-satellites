"""remote-satellites — remove satellite/plane trails from rotating star-field
timelapses.

The sky in a fixed-tripod star-field timelapse rotates about the celestial
pole. Trails (satellites, planes, meteors) are transient streaks that cut
across that rotation. remote-satellites estimates the rotation, "unwinds" it so the
stars sit still, runs a temporal median that rejects the transient streaks
while preserving the (now stationary) stars, then re-applies the rotation.
"""

__version__ = "0.1.0"
