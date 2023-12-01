""""
This file contains some predefined geometries for ease of use
"""

import numpy as np
from model.general.create_geometry import WALLDOWN, RIVER, WALLUP, SEA


############################### Geometry data ###########################
# Geometries defined in terms of points. Useful for geometries with straight sides.


def rectangle(B, L, Lextension=0):
    """
    Rectangular domain.

    Rectangular domain of length L+Lextension and width B

        y=B/2 |---------------------------|-----|
              |                           |     |
    (Sea) y=0 |                           |     | (River)
              |                           |     |
       y=-B/2 |---------------------------|-----|
             x=0                         x=L   x=L+Lextension
    """
    if Lextension==0:
        geometrydata = np.array(
            [[0, -B / 2, WALLDOWN], [L, -B / 2, RIVER], [L, B / 2, WALLUP],
             [0, B / 2, SEA]])
    else:
        geometrydata = np.array(
            [[-Lextension, -B / 2, WALLDOWN], [L, -B / 2,RIVER], [L, B / 2, WALLUP], [-Lextension, B / 2, SEA]])
    return geometrydata


def trapezoid(B, L1, L2):
    """
    Trapezoid domain.

    A trapezoidal domain with nonparallel sides at the Sea and River boundaries.
    The lower parallel side has length L1, the upper parallel L2 and the width is B.
    The sloped sides have the same slope.

                       <-------L2------>
                     x=(L1-L2)/2        x=(L1+L2)/2
                  y=B |------------------|
                    |                     |
    (Sea)         |                        |    (River)
                |                           |
          y=0 |------------------------------|
             x=0                            x=L1
    """

    geometrydata = np.array(
            [[0, 0, WALLDOWN], [L1, 0, RIVER], [(L1+L2)/2, B, WALLUP],
             [(L1-L2)/2, B, SEA]])
    return geometrydata




###################################### geometrycurves #########################################
# Geometries defined in terms of parametric curves. Useful for geometries with curved sides.

def exponential(B0, L, Lc):
    """
    Exponential domain.

    Exponential convergent domain with initial width B0, width convergence length Lc and length L.
    The spatially varying width is given by B(x) = B0 * exp(-x/Lc) for 0<x<L

        y=B0/2 |----\
               |     \-------\
               |              \------------|
    (Sea)  y=0 |                           | (River)
               |              /------------|
               |     /-------/
       y=-B0/2 |----/
             x=0                         x=L
    """

    def expside(t, B0, Lc, L1, L2, isUpper):
        """ Parametrised exponential side with initial width B0 and convergence length Lc. """
        xtilde = L1 + (L2 - L1) * t
        if isUpper:
            factor = 1
        else:
            factor = -1
        return np.array([xtilde, factor * B0 / 2 * np.exp(-xtilde / Lc)])

    def bottomexp(t):
        return expside(t, B0, Lc, 0, L, False)

    def topexp(t):
        return expside(t, B0, Lc, L, 0, True)


    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def rightside(t):
        return linearly_connect(t, bottomexp(1), topexp(0))

    def leftside(t):
        return linearly_connect(t, topexp(1), bottomexp(0))

    geometrycurves = [[bottomexp, WALLDOWN], [rightside, RIVER], [topexp, WALLUP], [leftside, SEA]]

    return geometrycurves


def exponential_rational(C1, C2, B0, L):
    """
    Exponential rational domain.

    Exponential rational convergent domain where the width is dependent on the composition between a rational and exponential function.

    The polynomials are constructed as in numpy.polyval, e.g.,
    The numerator of the rational function is constructed for N1 = len(C1) as:
        n(x) =  C1[0]*x^(N1-1) + C1[1]*x^(N1-2) ... + C1[N1-1]
    The denominator of the rational function is constructed for N2 = len(C2) as:
        d(x) =  C2[0]*x^(N2-1) + C2[1]*x^(N2-2) ... + C2[N2-1]
    The rational function is then given by
        r(x) = n(x)/d(x)

    The spatially varying exponential rational width reads
        B(x) = B0 * exp(-r(0)) * exp(r(x)),      for 0<x<L,
    with initial width B0.

        y=B0/2 |--\
               |    \---\
               |         \----\
               |               \--------------|
    (Sea)  y=0 |                              | (River)
               |                /-------------|
               |          /----/
               |    /---/
       y=-B0/2 |--/
              x=0                            x=L
    """
    def exp_rat_side(t, x1, x2, isUpper):
        """ Parameterised side of the rational exponential domain. The side represents the half width."""
        xtilde = x1 + (x2 - x1) * t
        if isUpper:
            factor = 1
        else:
            factor = -1

        def side(xtilde):
            def rational(xtilde):
                return np.polyval(C1, xtilde)/np.polyval(C2, xtilde)

            return factor * B0 * np.exp(-rational(0)) / 2 * np.exp(rational(xtilde))

        return np.array([xtilde, side(xtilde)])


    def bottomside(t):
        return exp_rat_side(t, 0, L, False)

    def topside(t):
        return exp_rat_side(t, L, 0, True)

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def rightside(t):
        return linearly_connect(t, bottomside(1), topside(0))

    def leftside(t):
        return linearly_connect(t, topside(1), bottomside(0))

    geometrycurves = [[bottomside, WALLDOWN], [rightside, RIVER], [topside, WALLUP], [leftside, SEA]]

    return geometrycurves


def annulus(r1, r2, theta1, theta2, Lextension=0):
    """
    Extended annular domain.

    Annular domain from theta1 to theta2 with inner radius r1 and outer radius r2.
    If Lextension==0, then returns normal annulus else an extended annulus is returned.

             _theta=theta2
             _____________
    (River)  |   |        \
             |___|___       \
     Lextension^     \       |
                      |_____ | _theta=theta1
                     r=r1    r=r2
                        (Sea)
    """

    def arc(t, r, theta1, theta2):
        """"Arc from theta1 to theta2 with radius r"""
        return np.array([r * np.cos((theta2 - theta1) * t + theta1), r * np.sin((theta2 - theta1) * t + theta1)])

    def arc1(t):
        return arc(t, r1, theta2, theta1)

    def arc2(t):
        return arc(t, r2, theta1, theta2)

    p2extension = np.array([-Lextension, arc2(1)[1]])
    p1extension = np.array([-Lextension, arc1(0)[1]])

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def bottom(t):
        return linearly_connect(t, arc1(1), arc2(0))

    def arc2_extended(t):
        return linearly_connect(t, arc2(1), p2extension)

    def side(t):
        return linearly_connect(t, p2extension, p1extension)

    def arc1_extended(t):
        return linearly_connect(t, p1extension, arc1(0))

    def side_arcs(t):
        return linearly_connect(t, arc2(1), arc1(0))


    if Lextension == 0:
        geometrycurves = [[arc1, WALLDOWN], [bottom, RIVER], [arc2, WALLUP], [side_arcs, SEA]]
    else:
        geometrycurves = [[arc1, WALLDOWN], [bottom, RIVER], [arc2, WALLUP], [arc2_extended, WALLUP], [side, SEA],
                        [arc1_extended, WALLDOWN]]
    return geometrycurves


def linearly_converging(r1, r2, theta, isSeaBoundaryCurved=True, isRiverBoundaryCurved=True):
    """
    linearly converging channel with possibly curved sides


    Linearly converging channel with angle _theta from r1 to r2

    _theta=_theta/2    /----\
                    /       \----\
                   |              \----|
            (Sea) |                   |   o  (River)
                   \              /----|
                    \      /----/
    _theta=-_theta/2   \----/
                    r=r1              r=r2

    """

    def arc(t, r, theta1, theta2):
        """"Arc from theta1 to theta2 with radius r"""
        return np.array([r * np.cos((theta2 - theta1) * t + theta1), r * np.sin((theta2 - theta1) * t + theta1)])

    def arc_sea(t):
        return arc(t, r1, np.pi - theta/2, np.pi + theta/2)

    def arc_river(t):
        return arc(t, r2, np.pi + theta / 2, np.pi - theta / 2)

    def linearly_connect(t, p1, p2):
        return p1 + (p2 - p1) * t

    def bottom(t):
        return linearly_connect(t, arc_sea(1), arc_river(0))

    def top(t):
        return linearly_connect(t, arc_river(1), arc_sea(0))

    # Straight sections
    def straight_sea(t):
        return linearly_connect(t, top(1), bottom(0))

    def straight_river(t):
        return linearly_connect(t, bottom(1), top(0))


    # Logic set sea and river boundaries
    if isSeaBoundaryCurved:
        # Curved sea boundary
        side_sea = arc_sea
    else:
        # Straight sea boundary
        side_sea = straight_sea

    if isRiverBoundaryCurved:
        #Curved river boundary
        side_river = arc_river
    else:
        # Straight river boundary
        side_river = straight_river


    geometrycurves = [[bottom, WALLDOWN], [side_river, RIVER], [top, WALLUP], [side_sea, SEA]]

    return geometrycurves


