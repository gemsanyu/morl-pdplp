import math
from math import radians, cos, sin, asin, sqrt

import folium
from folium.plugins import BeautifyIcon
import numpy as np


def haversine(coord1, coord2):

      R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(coord2[0] - coord1[0])
      dLon = radians(coord2[1] - coord1[1])
      lat1 = radians(coord1[0])
      lat2 = radians(coord2[0])

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      return R * c

def angleFromCoordinate(coord1, coord2):
    lat1, lat2 = coord1[0], coord2[0]
    long1, long2 = coord1[1], coord2[1]
    dLon = (long2 - long1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise

    return brng

def read_tour(solution_file_name, obj_file_name):
    tours_list = []
    with open(solution_file_name, "r") as solution_file:
        line = solution_file.readline()
        while(line):
            line = line.split()
            num_vec = int(line[1])
            tours = []
            for _ in range(num_vec):
                line_vec = solution_file.readline()
                line_vec = line_vec.split()
                tour = []
                for node in line_vec:
                    tour += [int(node)]
                tours += [tour]
            tours_list += [tours]
            # for _ in range(num_vec):
            #     line_vec = solution_file.readline()
            line = solution_file.readline()
    f_list = []
    with open(obj_file_name, "r") as obj_file:
        line = obj_file.readline()
        while(line):
            line = line.split()
            f = [float(line[0]), float(line[1])]
            f_list += [f]
            line = obj_file.readline()
        f_list = np.asanyarray(f_list)

    return f_list, tours_list

if __name__ == "__main__":
    num_nodes = 0
    coords = None
    instance_name = "bar-n100-2"
    graph_filename = instance_name+".txt"
    with open(graph_filename, "r") as graph_file:
        lines = graph_file.readlines()
        num_nodes = int(lines[0].split()[1])
        coords = np.zeros((num_nodes,2), dtype=np.float32)
        for i in range(1, num_nodes+1):
            strings = lines[i].split()
            idx = i-1
            coords[idx, 0], coords[idx,1] = float(strings[1]), float(strings[2])
    m = folium.Map(location=coords[0], zoom_start=12, tiles="OpenStreetMap")
    
    icon_star = BeautifyIcon(
        icon='house',
        inner_icon_style='color:purple;font-size:30px;border-color:red;',
        background_color='transparent',
        border_color='transparent',
    )
    folium.Marker(coords[0], icon=icon_star).add_to(m)
    for i in range(1,len(coords)):
        folium.Marker(coords[i]).add_to(m)
    
    colors = ["blue","green","red"]
    num_vec = 1
    model_name = "hnc-phn-po-init"
    solution_file_name = instance_name + "-" + str(num_vec) +"-"+ model_name + ".x"
    obj_file_name = instance_name + "-" + str(num_vec) +"-"+ model_name + ".y"
    
    f_list, tours_list = read_tour(solution_file_name, obj_file_name)
    min_idx = np.argmin(f_list, axis=0)
    # for tours in tours_list:
    #     print(tours)
    fg = folium.FeatureGroup("Lines")
    tours = tours_list[min_idx[0]]
    for i, tour in enumerate(tours):
        tour = tour + [0]
        tour_coords = coords[tour,:]
        folium.PolyLine(tour_coords, color=colors[i]).add_to(fg)
    # Adding arrowhead
    for i, tour in enumerate(tours):
        tour = tour + [0]
        tour_coords = coords[tour,:]
        print(tour)
        for j, tcoord in enumerate(tour_coords):
            if j == 0:
                continue
            prev_tcoord = tour_coords[j-1]
            lat_diff = tcoord[0]-prev_tcoord[0]
            lon_diff = tcoord[1]-prev_tcoord[1]
            rotation = angleFromCoordinate(prev_tcoord, tcoord)-90
            loc = 0.98*tcoord + 0.02*prev_tcoord
            folium.RegularPolygonMarker(location=loc, 
                                        fill_color=colors[i], 
                                        border_color=colors[i], 
                                        number_of_sides=3, 
                                        radius=10, 
                                        rotation=rotation, 
                                        fill=True,
                                        opacity=1,
                                        weight=0,
                                        fill_opacity=1).add_to(m)
    fg.add_to(m)
    folium.LayerControl(position="bottomright").add_to(m)
    m.save(graph_filename+"0.html")