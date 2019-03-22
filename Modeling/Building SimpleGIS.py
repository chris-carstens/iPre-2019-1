# Building SimpleGIS:
# - Python version: 
# - Author: Mauro S. Mendoza Elguera
# - Date: 2019-03-21

"""Learning Geospatial Analysis with Python P. 39"""

import turtle as t

NAME = 0
POINTS = 1
POP = 2  # Population count

state = ["COLORADO", [[-109, 37], [-109, 41], [-102, 41], [-102, 37]], 5187582]

cities = [["DENVER", [-104.98, 39.74], 634265],
          ["BOULDER", [-105.27, 40.02], 98889],
          ["DURANGO", [-107.88, 37.28], 17069]]

map_width = 400
map_height = 300

minx = 180
maxx = -180
miny = 90
maxy = -90

for x, y in state[POINTS]:
    if x < minx:
        minx = x
    elif x > maxx:
        maxx = x
    if y < miny:
        miny = y
    elif y > maxy:
        maxy = y

dist_x = maxx - minx
dist_y = maxy - miny
x_ratio = map_width / dist_x
y_ratio = map_height / dist_y


def convert(point):
    """

    :type point: list

    It transforms a point in the map coordinates from one of our data layers to
    pixel coordinates using the previous calculations.
    """
    lon = point[0]
    lat = point[1]
    x = map_width - ((maxx - lon) * x_ratio)
    y = map_height - ((maxy - lat) * y_ratio)

    # Python turtle graphics start in the middle of the screen so we must
    # offset the points so they are centered

    x = x - (map_width / 2)
    y = y - (map_height / 2)
    return [x, y]


t.up()
first_pixel = None
for point in state[POINTS]:
    pixel = convert(point)
    if not first_pixel:
        first_pixel = pixel
    t.goto(pixel)
    t.down()
t.goto(first_pixel)
t.up()
t.goto([0, 0])
t.write(state[NAME], align="center", font=("Arial", 16, "bold"))

for city in cities:
    pixel = convert(city[POINTS])
    t.up()
    t.goto(pixel)
    # Place a point for the city
    t.dot(10)
    # Label the city
    t.write(city[NAME] + ", Pop.: " + str(city[POP]), align="left")
    t.up()

biggest_city = max(cities, key=lambda city: city[POP])
t.goto(0, -200)
t.write("The biggest city is: " + biggest_city[NAME])

western_city = min(cities, key=lambda city: city[POINTS])
t.goto(0, -220)
t.write("The western-most city is: " + western_city[NAME])

t.pen(shown=False)
t.done()
