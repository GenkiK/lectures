def bilinear_interp(image, points):
    """Looks up the pixel values in an image at a given point using bilinear
    interpolation."""

    # Compute the four integer corner coordinates (top-left/right,
    # bottom-left/right) for interpolation, as well as the fractional part of
    # the coordinates.

    # Interpolate between the top two pixels.

    # Interpolate between the bottom two pixels.

    # Return the result of the final interpolation between top and bottom.

    # pointsはおそらく[x, y]の列．x, yはfloatなので，これを補間するtop-left/right, bottom-left/rightを求めて，それらでpointの点の色を補間する
    # imageは (height x width x 3)
    new_points = []
    for point in points:
        top_left = np.round(point)
        top_right = top_left + np.array([1, 0])
        bottom_left = top_left + np.array([0, 1])
        bottom_right = top_left + np.array([1, 1])
        a, b = point - top_left  # 横・縦の比
        color_vector = (
            a * b * image[top_left[0], top_left[1]]
            + (1 - a) * b * image[top_right[0], top_right[1]]
            + a * (1 - b) * image[bottom_left[0], bottom_left[1]]
            + (1 - a) * (1 - b) * image[bottom_right[0], bottom_right[1]]
        )

    return np.array(new_points)
