from shapely.geometry import box


class Bbox:

    def __init__(self, x0: float, y0: float, x1: float, y1: float, y_up: bool):
        self.minx = min(x0, x1)
        self.maxx = max(x0, x1)
        self.miny = min(y0, y1)
        self.maxy = max(y0, y1)
        self.y_up = y_up

    def __repr__(self):
        return f"Bbox({self.minx}, {self.miny}, {self.maxx}, {self.maxy})"

    def __str__(self):
        return f"({self.minx}, {self.miny}), ({self.maxx}, {self.maxy})"

    def width(self):
        return self.maxx - self.minx

    def height(self):
        return self.maxy - self.miny

    def to_shapely(self):
        return box(*self.to_tuple())

    def to_int(self):
        return BboxInt(self.minx, self.miny, self.maxx, self.maxy)

    def to_tuple(self):
        if self.y_up:
            return self.minx, self.miny, self.maxx, self.maxy
        else:
            return self.minx, self.maxy, self.maxx, self.miny

    def buffer(self, x: float, y: float):
        return self.__class__(
            self.minx - x, self.miny - y, self.maxx + x, self.maxy + y, self.y_up
        )


class BboxInt(Bbox):

    def __init__(self, x0: int, y0: int, x1: int, y1: int, y_up: bool):
        super().__init__(x0, y0, x1, y1, y_up)
        self.minx = int(self.minx)
        self.miny = int(self.miny)
        self.maxx = int(self.maxx)
        self.maxy = int(self.maxy)

    def folder_name(self):
        return f"{self.minx}_{self.miny}_{self.maxx}_{self.maxy}"

    def to_float(self):
        return Bbox(self.minx, self.miny, self.maxx, self.maxy, self.y_up)

    def buffer(self, x: int, y: int):
        return super().buffer(x, y)
