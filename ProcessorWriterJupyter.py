# Import and define useful modules
import math
import os
import pandas as pd
from shapely.geometry import GeometryCollection,LineString, Point, MultiPoint

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier
import copy

np.seterr(invalid='ignore')  # Suppress divide by zero error


def build_settings(filedir, filename, fileout, d, x_offset, y_offset, bed_temperature, floor, z_min, roof, f_print,
                   E_clean):
    settings = {
        'filedir': filedir,
        'filename': filename,
        'fileout': fileout,
        'd': d,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'bed_temperature': bed_temperature,
        'floor': floor,
        'z_min': z_min,
        'roof': roof,
        'f_print': f_print,
        'E_clean': E_clean
    }
    return settings


def cartesian2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def cartesian3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def coordinater_wkt(string_in, idx):
    # Takes a set of coordinates in string form, where each coordinate is separated by a space
    # Rounds it to 3 decimal place, and returns it as a list of floats
    # Input - string_in - list of strings, each of which is a set of coordinates
    # Output - [x, y, z, e_id] - list of floats where e_id is element ID
    form = []
    # Now run through a make x- and y- coordinates
    for i in range(0, len(string_in)):
        fragment = string_in[i].split()
        x = round(float(fragment[0]), 3)
        y = round(float(fragment[1]), 3)
        z = 0.0
        e_id = idx
        form.append([x, y, z, e_id])
    return form


def distance_calculator(df):
    # Drop NaN values (usually the first value)
    dx = df['x'].diff()
    dy = df['y'].diff()
    dz = df['z'].diff()

    # Calculate the Euclidean distance between consecutive rows
    distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    df['distance_from_last'] = distances
    return df
def select_min_key(d):
    # Step 1: Keys with odd-length lists
    odd_keys = [k for k in d if len(d[k]) % 2 == 1]

    if odd_keys:
        # Return the key with the shortest odd-length list
        return min(odd_keys, key=lambda k: len(d[k]))
    else:
        # Step 2: If no odd-length lists, get keys with even-length lists
        even_keys = [k for k in d if len(d[k]) % 2 == 0]
        if even_keys:
            return min(even_keys, key=lambda k: len(d[k]))
        else:
            return None  # In case all lists are empty or something went wrong

def vector_calculator(line_string, intersection):
    line_length =len( line_string.coords)
    # print("linelengtht",line_length)
    tenth_of_line = int(line_length/10)
    # print("tenth of line",tenth_of_line)
    target_coordinates = intersection
    coords_list = list(line_string.coords)
    closest_index = min(
        range(len(coords_list)),
        key=lambda i: Point(coords_list[i]).distance(intersection)
    )
    start_coord_index=closest_index
    if start_coord_index + tenth_of_line < line_length:
        end_coord_index = start_coord_index + tenth_of_line
    else:
        end_coord_index = max(0, start_coord_index - tenth_of_line)
    x1, y1 = line_string.coords[start_coord_index]


    x2, y2 = line_string.coords[end_coord_index]
    v = np.array([x2-x1,y2-y1])
    # print("x1", x1, "x2",x2,"y1",y1,"y2",y2)
    # print("vector coordinates", v)
    return v
def angle_calculator(line_string_1, line_string_2, intersection):
    v1 = vector_calculator(line_string_1, intersection)
    v2 = vector_calculator(line_string_2, intersection)

    v1_norm = v1/np.linalg.norm(v1)
    v2_norm = v2/ np.linalg.norm(v2)
    # print("v1 normalised", v1_norm)
    # print("v2 normalised", v2_norm)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    # print(dot_product, "dot product")
    cross_product = np.cross(v1_norm, v2_norm)
    radian_angle = np.arctan2(cross_product,dot_product)
    degree_angle = np.degrees(radian_angle)
    return degree_angle
def interpolated_z(intersection_point, df):

    df["xy_distance"] = np.sqrt((df['x'] - intersection_point.x)**2 + (df['y'] - intersection_point.y)**2)

    closest_row = df.loc[df["xy_distance"].idxmin()]
    # print(closest_row)

    target_index = df["xy_distance"].argmin() # the row you're interested in
    window = 5  # number of rows before and after

    # Slice around the target
    start = max(0, target_index - window)
    end = min(len(df), target_index + window + 1)

    # Show the rows
    # print("this is the data frame around the minimum distance",df.iloc[start:end])
    z_value = closest_row["z"]
    return z_value


def z_tenth_from_intersection_calculator(intersection, df, line_string):
    line_length = len(line_string.coords)

    tenth_of_line = int(line_length / 10)

    target_coordinates = intersection
    coords_list = list(line_string.coords)
    closest_index = min(
        range(len(coords_list)),
        key=lambda i: Point(coords_list[i]).distance(intersection)
    )
    start_coord_index = closest_index
    if start_coord_index + tenth_of_line < line_length:
        end_coord_index = start_coord_index + tenth_of_line
    else:
        end_coord_index = max(0, start_coord_index - tenth_of_line)
    z_point = Point(line_string.coords[end_coord_index])
    print(z_point, "zpoint")
    z_value = interpolated_z(z_point, df)
    return z_value

def validator(unprinted_lines, df):
    """"Checks if the printing of each line affects the printing of future lines and returns those that are able to be printed without affecting the printing of subsequent lines"""

    valid_lines = []
    # print(f"checking {len(unprinted_lines)}")
    for line_id_test in unprinted_lines:
        print(f"test line{line_id_test}")
        is_valid = True  # assume valid until proven otherwise

        current_line_coords = df[df["line_id"] == line_id_test][["x", "y"]]
        current_line_xyz = df[df["line_id"] == line_id_test][["x", "y", "z"]]
        current_line_linestring = LineString(current_line_coords.to_numpy())

        if len(current_line_coords) < 2:
            print(f"line skipped too short{line_id_test}")
            continue  # skip invalid lines

        for line_id in unprinted_lines:
            print(f"compared line {line_id}")
            if line_id == line_id_test:
                print("line id == line id test")
                continue

            compared_line_coords = df[df["line_id"] == line_id][["x", "y"]]
            compared_line_xyz = df[df["line_id"] == line_id][["x", "y", "z"]]
            print("compared line coords generated")

            if len(compared_line_coords) < 2:
                print(f"compared line too short line id {line_id}")
                continue

            compared_line_linestring = LineString(compared_line_coords.to_numpy())
            print("compared line string generated")

            if current_line_linestring.intersects(compared_line_linestring):
                intersection = current_line_linestring.intersection(compared_line_linestring)

                print(f"test line{line_id_test}intersects with line {line_id} at {intersection}")
                if isinstance(intersection, GeometryCollection):
                    for geometry in intersection.geoms:
                        if geometry.geom_type == "LineString":
                            try:
                                current_z_atparallel = current_line_xyz.iloc[min(20, len(current_line_xyz)-1)]["z"]
                                compared_z_atparallel = compared_line_xyz.iloc[min(20, len(compared_line_xyz)-1)]["z"]
                                print(f"current line z at linestring(parallel intersection){current_z_atparallel} compared z at angle{compared_z_atparallel}")
                            except IndexError:
                                print("index error occured")
                                continue  # skip if not enough points

                            if current_z_atparallel > compared_z_atparallel + 0.03:
                                is_valid = False
                                print(f"current z above compared z")
                                break

                if intersection.geom_type == "Point":
                    z_current = interpolated_z(intersection, current_line_xyz)
                    z_compare = interpolated_z(intersection, compared_line_xyz)
                    print(f"zvalues of current{z_current} and compare{z_compare}")
                    if np.isclose(z_current, z_compare, rtol=2e-02, atol=1e-08):
                        angle = angle_calculator(current_line_linestring, compared_line_linestring, intersection)
                        print(f"angle between lines{angle} test line ={line_id_test} compared line = {line_id}")
                        if (0.02 < angle < 25) or (-0.02> angle> -25):
                            try:
                                current_z_at20 = z_tenth_from_intersection_calculator(intersection, df, current_line_linestring)
                                compared_z_at20 = z_tenth_from_intersection_calculator(intersection, df, compared_line_linestring)
                                print(f"current line z at angle{current_z_at20} compared z at angle{compared_z_at20}")
                            except IndexError:
                                print("index error occured")
                                continue  # skip if not enough points

                            if current_z_at20 >= compared_z_at20 + 0.03:
                                is_valid = False
                                print(f"current z above compared z")
                                break
                        else:
                            continue  # no disqualification
                            print("not at node")
                    else:
                        if z_current >= z_compare:
                            is_valid = False
                            print("z value of current line is above compared line")
                            break

                elif intersection.geom_type == "MultiPoint":
                    print("multipoint")
                    for pt in intersection.geoms:
                        z_current = interpolated_z(pt, current_line_xyz)
                        print(z_current,"z_current")
                        z_compare = interpolated_z(pt, compared_line_xyz)
                        print(z_compare, "z_compare")
                        if z_current > z_compare + 0.07:
                            is_valid = False
                            print("not valid")
                            break
                    if not is_valid:
                        print("random thing")
                        break
            else:
                print("final else")
                is_valid = True

        if is_valid:
            valid_lines.append(line_id_test)
    print("valid lines", valid_lines)
    return valid_lines
def node_connectivity_finder(dictionary):
    new_dictionary = {}

    for key1, values1 in dictionary.items():
        shared_keys = []

        for key2, values2 in dictionary.items():
            if key1 == key2:
                continue

            # Count how many values from values1 are also in values2
            shared_count = sum(1 for v in values1 if v in values2)

            # Repeat key2 shared_count times
            shared_keys.extend([key2] * shared_count)

        new_dictionary[key1] = shared_keys


    return new_dictionary
def line_from_nodes(start_node, end_node,valid_line_cluster_dict):
    end_node_list= valid_line_cluster_dict[end_node]
    start_node_list = valid_line_cluster_dict[start_node]
    shared_lines = [item for item in start_node_list if item in end_node_list]
    return shared_lines[0]

def remove_line(valid_line_cluster_dict, start_node, next_node, line):
    valid_line_cluster_dict[start_node].remove(line)
    valid_line_cluster_dict[next_node].remove(line)

def remove_node(valid_node_link_dict, u, v):
    valid_node_link_dict[u].remove(v)
    valid_node_link_dict[v].remove(u)

def depth_first_search(next_node, valid_node_link_dict, visited):
    visited[next_node] = True

    for neighbor in valid_node_link_dict[next_node]:
        if not visited[neighbor]:
            depth_first_search(neighbor, valid_node_link_dict, visited)

def bridge_check(next_node, start_node, valid_node_link_dict,node_number):
    if len(valid_node_link_dict[start_node]) == 1:
        return True, 0
    node_link_copy = copy.deepcopy(valid_node_link_dict)
    visited = {key:False for key in node_link_copy}
    count1 = 0
    depth_first_search(next_node, node_link_copy, visited)
    count1 = sum(1 for value in visited.values() if value is True)

    remove_node(node_link_copy, start_node, next_node)

    visited = {key:False for key in node_link_copy }
    count2 = 0
    depth_first_search(start_node, node_link_copy, visited)
    count2 = sum(1 for value in visited.values() if value is True)

    node_link_copy[start_node].append(next_node)
    node_link_copy[next_node].append(start_node)
    print("count 1", count1, "count2", count2)
    print(f"Returning from bridge_check: {(count1 == count2, count2)}")
    return (count1 == count2), count2
def recursive_eulerian(node_path, edges, start_node, valid_node_link_dict, node_number, valid_line_cluster_dict):
    if not valid_node_link_dict.get(start_node):
        print(f"No more connections from {start_node}. Ending recursion.")
        return
    any_bridge_passed = False

    count2_map = {}

    for node in valid_node_link_dict[start_node]:
        print(valid_node_link_dict[start_node], "valid node connections dictionary")
        next_node= node
        print(next_node, "next node")
        passed_bridge, count2 = bridge_check(next_node, start_node, valid_node_link_dict, node_number)
        if count2 is not None:
            count2_map[next_node] = count2
        if passed_bridge:
            any_bridge_passed = True
            if start_node == node_path[-1]:
                print(node_path, "node path if passed bridge")
                print(start_node, "start node", next_node, "next node")
                line = line_from_nodes(start_node, next_node, valid_line_cluster_dict)
                edges.append(line)
                print("appending line", line)
                node_path.append(next_node)
                remove_node(valid_node_link_dict, start_node, next_node)
                remove_line(valid_line_cluster_dict, start_node, next_node, line)
                #repeat with start node as the next node
                recursive_eulerian(node_path, edges, next_node, valid_node_link_dict, node_number, valid_line_cluster_dict)
                break
    if not any_bridge_passed:
        if count2_map:
            next_node = min(count2_map, key=count2_map.get)
        else:

            next_node = valid_node_link_dict[start_node][0]
        if start_node == node_path[-1]:
            print(node_path, "node path from no bridge passed")
            print(edges, "edges from no bridge passed")
            line = line_from_nodes(start_node, next_node, valid_line_cluster_dict)
            print(start_node, "start node", next_node, "end node from no bridge passed")
            edges.append(line)
            print("appending line", line)
            node_path.append(next_node)
            remove_node(valid_node_link_dict, start_node, next_node)
            remove_line(valid_line_cluster_dict, start_node, next_node, line)
            # repeat with start node as the next node
            recursive_eulerian(node_path, edges, next_node, valid_node_link_dict, node_number, valid_line_cluster_dict)


def fleurys_algorithm(clusters, cluster_dictionary, connectivity_dictionary, valid_lines, edges_grouped, node_path_grouped):
    valid_line_cluster_dict= {}

    for cluster, lines in cluster_dictionary.items():
        valid_line_cluster_dict[cluster] = [item for item in lines if item in valid_lines]

    valid_node_link_dict = node_connectivity_finder(valid_line_cluster_dict)
    print(valid_node_link_dict)

    while valid_line_cluster_dict:
        node_number = len(valid_node_link_dict)

        filtered = {k: v for k, v in valid_node_link_dict.items() if v not in ('', None,[])}
        print(filtered)
        if filtered:
            min_key = select_min_key(filtered)

        else:
            print("No valid entries")
            break
        start_node = min_key
        print(start_node, "start node")
        node_path = []
        edges = []
        node_path.append(start_node)
        recursive_eulerian(node_path, edges, start_node, valid_node_link_dict, node_number, valid_line_cluster_dict)
        print(edges, "edges")
        print(node_path, "node path")
        edges_grouped.append(edges)
        node_path_grouped.append(node_path)

    print(edges_grouped)
    print(node_path_grouped)






def eulerficator(df, terminal_points, nodes):
    terminal_points_nogroups = terminal_points.reset_index()
    clusters = terminal_points_nogroups['cluster'].unique()
    lines = terminal_points_nogroups['line_id'].unique()
    # print(terminal_points)
    # print("----------")
    # print(terminal_points_nogroups)
    # print(nodes)
    # print(df)
    # Run through each cluster and create two dictionaries
    # 1 - cluster numbers as key and the connecting lines as values
    # 2 - cluster numbers as key and the number of connecting nodes as values
    cluster_dict = {}
    connectivity_dict = {}

    for c in clusters:
        connecting_lines = terminal_points_nogroups[terminal_points_nogroups['cluster'] == c]
        cluster_dict[c] = list(connecting_lines['line_id'].values)

        connectivity_dict[c] = len(connecting_lines)
    print("cluster_dict")
    print(cluster_dict)
    print("connectivity_dict")
    print(connectivity_dict)
    print("terminal_points_nogroups")
    print(terminal_points_nogroups)
    print("terminal_points")
    print(terminal_points)
    print("clusters")
    print(clusters)
    print("lines")
    print(lines)




    edges_grouped = []

    node_path_grouped = []
    while sum(len(sublist) for sublist in edges_grouped) < len(lines):


        # Sort lines by height order - bottom up
        min_z = df.groupby('line_id')['z'].min()
        heightsorted_line_ids = min_z.sort_values().index.tolist()
        edges_ungrouped = flattened_edges = [item for sublist in edges_grouped for item in sublist]
        unprinted_lines = [n for n in heightsorted_line_ids if n not in edges_ungrouped]
        # print(df)
        ### where the validator goes###

        valid_lines = validator(unprinted_lines, df)
        if not valid_lines:
            print("No valid lines found. Breaking to avoid infinite loop.")
            break  # <- prevents infinite loop

        fleurys_algorithm(clusters, cluster_dict, connectivity_dict, valid_lines, edges_grouped, node_path_grouped)

        while len(valid_lines)>0:
            next_line = valid_lines[0]
            valid_lines = valid_lines[1:]  # Remove the printed line from the list of remaining lines
            unprinted_lines.remove(next_line)


    node_order = [item for sublist in node_path_grouped for item in sublist]
    print(node_order,"node order")
    line_order = [item for sublist in edges_grouped for item in sublist]
    print(line_order, "line order")
    line_order_grouped = edges_grouped
    node_order_grouped = node_path_grouped
    node_plotter(df, terminal_points)
    return line_order, node_order, line_order_grouped, node_order_grouped

def e_calculator(df):
    alpha = 1
    diameter = 1
    df['E'] = np.pi * alpha * df['distance_from_last'] * (diameter / 2) ** 2  # Amount to extrude
    return df


def inkscape_preprocess(data):
    # Split wkt file into LINESTRING/POLYGON elements, return x and y coordinates
    # Each element corresponds to a different element of the wkt
    # Note the coordinates strings are sometimes so long print won't show them all
    print('inkscape file being processed')
    pattern = []
    # Each element is a LINESTRING or POLYGON
    for idx, element in enumerate(data):
        element = wkt_splitter(element)
        element = coordinater_wkt(element, idx)
        if not len(element):
            print('Ignoring point')
        if len(element):
            pattern.append(element)

    pattern = [item for sublist in pattern for item in sublist]
    pattern = pd.DataFrame(pattern, columns=['x', 'y', 'z', 'line_id'])
    return pattern


def line_order_corrector(df, line_order, line_order_grouped, nodes, node_order_grouped):
    # Make lines run in node order.
    df = df.set_index('line_id').loc[line_order].reset_index()  # Reorder the dataframe based on the line order

    for idx_path, path in enumerate(line_order_grouped):
        for idx_line, line in enumerate(path):
            start_node = node_order_grouped[idx_path][idx_line]
            line_start = df[df['line_id'] == line].iloc[0][['x', 'y', 'z']].values
            node_loc = nodes.loc[start_node][['x', 'y', 'z']].values

            if not np.array_equal(line_start, node_loc):
                # print('Reversing line ', line)
                new_line = df[df['line_id'] == line].iloc[::-1]
                df = df[df['line_id'] != line]
                df = pd.concat([df, new_line])

    df = df.set_index('line_id').loc[line_order].reset_index()  # Reorder the dataframe based on the line order
    df = distance_calculator(df)
    return df


def midlinejumpsplitter(shape):
    # This needs generalising to lines with more than 2 jumps
    # print('Splitting line with id:', shape['line_id'].unique()[0])
    id = shape['line_id'].unique()[0]
    # Split the line at the index - at the moment uses a completely arbitrary distance of 2mm
    split_index = shape[shape['distance_from_last'] > 2.].index[0]
    shape1 = shape.iloc[:split_index]
    shape1['line_id'] = 1
    shape1['distance_from_last'].iloc[0] = np.nan
    shape2 = shape.iloc[split_index:]
    shape2['line_id'] = 2
    shape2['distance_from_last'].iloc[0] = np.nan
    return shape1, shape2


def node_finder(df):
    # Use hierarchical clustering to note common start / end points
    # Calculate node positions based on cluster centroids
    # Append centroids to start / end of each line

    start_points = df.groupby('line_id').first()  # First point of each line
    end_points = df.groupby('line_id').last()  # Last
    terminal_points = pd.concat([start_points, end_points])  # Combine the two
    dist_mat = dist.pdist(terminal_points[['x', 'y', 'z']].values)
    link_mat = hier.linkage(dist_mat)
    # fcluster assigns each of the particles in positions a cluster to which it belongs
    cluster_idx = hier.fcluster(link_mat, t=1,
                                criterion='distance')  # t defines the max cophonetic distance in a cluster
    terminal_points['cluster'] = cluster_idx

    # Calculate the mean position of each cluster
    nodes = terminal_points.groupby('cluster').mean()
    for n in terminal_points.index.unique():
        clusters = terminal_points.loc[n]['cluster']
        for c in clusters:
            new_point = nodes.loc[c:c]
            line = df[df['line_id'] == n]
            line_start = line.head(1)
            line_end = line.tail(1)
            new_point[['r', 'g', 'b', 'line_id']] = line_start[['r', 'g', 'b', 'line_id']].values

            start_sep = dist.euclidean(new_point[['x', 'y', 'z']].values[0], line_start[['x', 'y', 'z']].values[0])
            end_sep = dist.euclidean(new_point[['x', 'y', 'z']].values[0], line_end[['x', 'y', 'z']].values[0])
            if start_sep < end_sep:
                line = pd.concat([new_point, line])
            elif start_sep > end_sep:
                line = pd.concat([line, new_point])

            df = df[df['line_id'] != n]
            df = pd.concat([df, line])
            df = df.reset_index(drop=True)
            df = distance_calculator(df)

            # Set distance from last to NaN for the first row of each line
            df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan
    return df, terminal_points, nodes


def node_plotter(df, terminal_points):
    # Plots lines assigning a colour to each line_id
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot and label lines
    for n in np.unique(df['line_id'].values):
        line_data = df[df['line_id'] == n]
        ax.plot(line_data['x'], line_data['y'], line_data['z'])

        # Add label at the midpoint of the line
        mid_index = len(line_data) // 2
        x_mid = line_data['x'].values[mid_index]
        y_mid = line_data['y'].values[mid_index]
        z_mid = line_data['z'].values[mid_index]

        ax.text(x_mid, y_mid, z_mid, str(n), fontsize=9, color='black')

    # Plot and label terminal points
    for n in np.unique(terminal_points['cluster'].values):
        cluster_data = terminal_points[terminal_points['cluster'] == n]
        ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'])

        for _, row in cluster_data.iterrows():
            ax.text(row['x'], row['y'], row['z'], str(row['cluster']),
                    fontsize=8, color='red', zorder=10)

    ax.view_init(elev=30, azim=60)
    plt.show()


def remove_overlap(shape):
    # Checks for any large jumps at the end of a line and, if found, moves the last row to the top
    # This hopefully removes the jump and makes the line continuous
    # Note this seems very fragile and I probably need add a second check for large jumps at the end of the code
    if shape.iloc[-1]['distance_from_last'] > 2.:
        # Move the last row to the top
        last_row = shape.iloc[[-1]]  # Select the last row as a DataFrame
        remaining_rows = shape.iloc[:-1]  # Select all rows except the last
        shape = pd.concat([last_row, remaining_rows]).reset_index(drop=True)

    shape = distance_calculator(shape)
    return shape


def preprocess(df, settings):
    # Calculate the distance between consecutive points
    df = distance_calculator(df)
    # If line ID column doesn't exist, assign line IDs based on RGB values
    if 'line_id' not in df.columns:
        df['line_id'] = pd.factorize(df[['r', 'g', 'b']].apply(tuple, axis=1))[0]
    # Remove large jumps at the end of lines
    df = df.groupby('line_id', group_keys=False).apply(remove_overlap)

    # Recalculate the distance between consecutive points
    df = distance_calculator(df)

    # Set distance from last to NaN for the first row of each line
    df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan

    # Find and split lines that have big jumps in the middle, e.g., inlet and outlet lines
    df = shapesplitter(df)

    # Set distance from last to NaN for the first row of each line
    df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan

    df['x'] = df['x'] - df['x'].mean() + settings['x_offset']
    df['y'] = df['y'] - df['y'].mean() + settings['y_offset']

    return df


def shape_prep(settings):
    # Load pattern data and convert to DataFrame
    df = pd.read_csv(os.path.join(settings['filedir'], settings['filename']), header=None)
    df.columns = ['x', 'y', 'z', 'r', 'g', 'b']

    df = preprocess(df, settings)

    # Find and plot nodes via hierarchical clustering
    df, terminal_points, nodes = node_finder(df)
    node_plotter(df, terminal_points)

    line_order, node_order, line_order_grouped, node_order_grouped = eulerficator(df, terminal_points, nodes)

    # Correct line order to run in node order
    df = line_order_corrector(df, line_order, line_order_grouped, nodes, node_order_grouped)
    return df, line_order_grouped


def shapesplitter(df):
    # Identify line IDs that have large jumps in the middle
    line_ids = np.sort(df['line_id'].unique())  # Sorting makes life easier later
    line_ids_new = line_ids.copy()  # A list of line IDs that we're going to update
    for line_id in line_ids:
        shape = df[df['line_id'] == line_id]
        if shape['distance_from_last'].max() > 2.:
            shape1, shape2 = midlinejumpsplitter(shape)
            df = df[df['line_id'] != line_id]
            line_ids_new = line_ids_new[line_ids_new != line_id]
            line_ids_new = np.append(line_ids_new, [line_ids_new[-1] + 1, line_ids_new[-1] + 2])
            shape1['line_id'], shape2['line_id'] = line_ids_new[-2], line_ids_new[-1]
            shape = pd.concat([shape1, shape2])
            df = pd.concat([df, shape])
    return df
import numpy as np



def gcode_writer(df, settings, line_order_grouped):
    # Calculate extrusion amount between points
    df_print = e_calculator(df, settings, line_order_grouped)
    # Offset path so head doesn't crash into print bed
    df_print['z'] = df_print['z'] + settings['z_min']
    df_print = df_print.round(3)   # 3dp max
    # Write sections to file
    preamble(settings)
    cleaning(settings)
    for path in line_order_grouped:
        position_printhead(df_print, path[0], settings)
        for line_id in path:
            print_line(df_print, line_id, settings)
        raise_printhead(df_print, settings)
    postamble(settings)

    return df_print


def e_calculator(df, settings, line_order_grouped):
    d = settings['d']  # Nozzle diameter
    alpha = 0.7034   # Extrusion multiplier
    # Start of print code
    df = df[df['distance_from_last'] != 0.]
    df = df.fillna(0)
    df['V_mL'] = np.pi*((d/2)**2)*df['distance_from_last']
    df['E'] = np.pi*(1/alpha)*df['V_mL']  # Amount to extrude

    # Set the extrusion amount for the first line of each path to zero
    for path in line_order_grouped:
        index_to_update = df[df['line_id'] == path[0]].index[0]  # Find the index of the first match
        df.loc[index_to_update, 'E'] = 0  # Update the value at that index

    df['E_cumulative'] = df['E'].cumsum()
    return df


def preamble(settings):
    preamble_out = """; Setup section
M82 ; absolute extrusion mode
G90 ; use absolute positioning
M104 S0.0 ; Set Hotend Temperature to zero
M140 S{} ; set bed temp
M190 S{} ; wait for bed temp
G28 ; home all
G92 E0.0 ; Set zero extrusion
M107 ; Fan off""".format(settings['bed_temperature'], settings['bed_temperature'])
    with open(settings['fileout'], "w") as file:
        file.write(preamble_out)


def cleaning(settings):
    cleaning_out = """\n
; Cleaning section
G1 F800 ; Set speed for cleaning
G1 X-50 Y50 ; Move to front left corner
G1 F500 ; Slow down to remove vibration
G1 Z{} ; Lower printhead to floor
G1 X50 Y50 E{} ; Move to front right corner
G1 Z{} ; Raise printhead
G1 X97.5 Y147 F2000 ; Move printhead to centre of printbed
G92 X0 Y0 E0 ; Set zero extrusion""".format(settings['floor'], settings['E_clean'], settings['roof'])
    with open(settings['fileout'], "a") as file:
        file.write(cleaning_out)


def postamble(settings):
    postamble_out = """\n
; End of print
M140 S0 ; Set Bed Temperature to zero
M107 ; Fan off
M140 S0 ; turn off heatbed
M107 ; turn off fan
G1 Z{} ; Raise printhead
G1 X178 Y180 F4200 ; park print head
G28 ; Home all
M84 ; disable motors
M82 ; absolute extrusion mode
M104 S0 ; Set Hotend Temperature to zero
; End of Gcode""".format(settings['roof'])
    with open(settings['fileout'], "a") as file:
        file.write(postamble_out)


def position_printhead(df, line_id, settings):
    first_line = df[df['line_id'] == line_id].iloc[0]
    positioning = """\n
; Initial positioning for new print path
G1 F800          ; Printhead speed for initial positioning
G1 X{} Y{}       ; XY-coords of first point of path
G1 Z{}           ; Z-coord of first point of path
G4 S2            ; Dwell for 2 seconds for karma / aligment
G1 F{}           ; Set printhead speed""".format(first_line['x'], first_line['y'], first_line['z'], settings['f_print'])
    with open(settings['fileout'], "a") as file:
        file.write(positioning)


def print_line(df, line_id, settings):
    line = df[df['line_id'] == line_id]
    start_line = "\n\n; Start of line number: " + str(line_id) + "\n"
    gcode_output = "\n".join(
        "G1 X" + line['x'].astype(str) + " Y" + line['y'].astype(str) + " Z" + line['z'].astype(str) + " E" + line['E_cumulative'].astype(str))
    with open(settings['fileout'], "a") as file:
        file.write(start_line)
        file.write(gcode_output)


def raise_printhead(df, settings):
    # Give 5 mm clearance
    raise_printhead_out = """\n
; Raise printhead
G1 Z{} F200""".format(5+df['z'].max())
    with open(settings['fileout'], "a") as file:
        file.write(raise_printhead_out)


# File settings
filedir = './Rhino/'  # Directory of the file you're loading
# Name of the output file
filename ='VesselBranched6.txt'   # Name of the file you're loading
fileout = 'Failure_Intersection_Test3.gcode'

# Print settings
d = 0.1  # Target fibril diameter in mm
x_offset, y_offset = 0, 0  # Offset of the print from the origin
bed_temperature = 0  # Bed temperature in degrees C
floor = -61  # z-coord at which nozzle hits printbed in mm
z_min = -58  # Minimum z-coord for the print in mm
roof = 0  # Height to retract printhead to at end of print
f_print = 200  # Printhead speed in mm / min
E_clean = 10
settings = build_settings(filedir, filename, fileout, d, x_offset, y_offset, bed_temperature, floor, z_min, roof, f_print,
                   E_clean)
df, line_order_grouped = shape_prep(settings)
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import numpy as np

# Load your DataFrame
# Replace this with your actual data loading
# Example format: df = pd.read_csv("your_data.csv")
# For demo, here's the column structure based on your sample


# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define colors (markers removed)
colors = plt.cm.tab10.colors

# Plot each line_id as a continuous line
for idx, (line_id, group) in enumerate(df.groupby('line_id')):
    color = colors[idx % len(colors)]
    ax.plot(group['x'], group['y'], group['z'],
            label=f'Line {line_id}',
            color=color,
            linewidth=2)  # Optional: make lines more visible

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Line Plot Grouped by line_id')
ax.legend()
plt.tight_layout()
plt.show()
