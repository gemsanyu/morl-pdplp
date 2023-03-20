import folium
import numpy as np

if __name__ == "__main__":
    num_nodes = 0
    coords = None
    graph_filename = "berlin.txt"
    with open(graph_filename, "r") as graph_file:
        lines = graph_file.readlines()
        num_nodes = int(lines[0].split()[1])
        coords = np.zeros((num_nodes,2), dtype=np.float32)
        for i in range(1, num_nodes+1):
            strings = lines[i].split()
            idx = i-1
            coords[idx, 0], coords[idx,1] = float(strings[1]), float(strings[2])
    print(num_nodes)
    print(coords)
    m = folium.Map(location=coords[0], zoom_start=12, tiles="Stamen Terrain")
    for i in range(500):
        folium.Marker(coords[i]).add_to(m)
    m.save(graph_filename+".html")