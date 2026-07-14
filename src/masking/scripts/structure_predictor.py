

"""bad structures
MD585.229.tif
MD585.253.tif
MD589.295.tif
"""




def create_coords(points):
    px = [a[0] for a in points]
    py = [a[1] for a in points]
    poly = [(x, y) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]
    x1 = int(np.min(px))
    y1 = int(np.min(py))
    x2 = int(np.max(px))
    y2 = int(np.max(py))
    w = x2 - x1
    h = y2 - y1
    return x1, x2, y1, y2, w, h
