import folium
from folium.plugins import BeautifyIcon
import numpy as np

if __name__ == "__main__":
    num_nodes = 0
    coords = None
    filename = "bar-n100-3.txt"
    instance_filename = "test/"+filename
    with open(instance_filename, "r") as instance_file:
        lines = instance_file.readlines()
        num_nodes = int(lines[4].split()[1])
        print(num_nodes)
        coords = np.zeros((num_nodes+2,2), dtype=np.float32)
        for i in range(11, num_nodes+11):
            strings = lines[i].split()
            idx = i-11
            coords[idx, 0], coords[idx,1] = float(strings[1]), float(strings[2])
    m = folium.Map(location=np.asanyarray([[41.380234, 2.145042]]), zoom_start=12, tiles="Stamen Terrain")
    num_request = int((num_nodes-1)/2)
    print(num_request)
    for i in range(num_nodes):
        if i == 0:
            depot_icon = BeautifyIcon(
                icon='home',
                inner_icon_style='color:orange;font-size:20px;',
                background_color='transparent', 
                border_color='transparent', 
            )
            folium.Marker(coords[i],icon=depot_icon).add_to(m)
        elif i < num_request:
            # square marker
            icon_square = BeautifyIcon(
                icon_shape='rectangle-dot',
                border_width=4,
                border_color='green'
            )
            folium.Marker(coords[i],icon=icon_square).add_to(m)
        else:          
            icon_circle = BeautifyIcon(
                icon_shape='circle-dot',
                border_width=4,
                border_color='blue'
            )
            folium.Marker(coords[i],icon=icon_circle).add_to(m)
        # folium.Marker(coords[i], ).add_to(m)
    m.save(filename+".html")