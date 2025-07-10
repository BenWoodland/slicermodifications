# Import and define useful modules
import math
import os
import pandas as pd
from shapely.geometry import LineString

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

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

def vector_calculator(line_string):
    x1, y1 = line_string.coords[0]
    x2, y2 = line_string.coords[1]
    v = np.array([x2-x1,y2-y1])
    return v
def angle_calculator(line_string_1, line_string_2):
    v1 = vector_calculator(line_string_1)
    v2 = vector_calculator(line_string_2)

    v1_norm = v1/np.linalg.norm(v1)
    v2_norm = v2/ np.linalg.norm(v2)

    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    radian_angle = np.arccos(dot_product)
    degree_angle = np.degrees(radian_angle)
    return degree_angle

def validator(unprinted_lines, df):
    """Returns lines that can be printed without affecting future lines."""
    valid_lines = []

    for line_id_test in unprinted_lines:
        current_line_coords = df[df["line_id"] == line_id_test][["x", "y"]]
        current_line_xyz = df[df["line_id"] == line_id_test][["x", "y", "z"]]
        current_line_linestring = LineString(current_line_coords.to_numpy())

        for line_id in unprinted_lines:
            if line_id == line_id_test:
                continue

            compared_line = df[df["line_id"] == line_id][["x", "y"]]
            compared_line_xyz = df[df["line_id"] == line_id][["x", "y", "z"]]
            if len(compared_line) < 2:
                continue

            compared_line_linestring = LineString(compared_line.to_numpy())

            if current_line_linestring.intersects(compared_line_linestring):
                intersection = current_line_linestring.intersection(compared_line_linestring)

                if intersection.geom_type == "Point":
                    intersection_x = intersection.x
                    z1 = current_line_xyz[current_line_xyz["x"] == intersection_x]["z"].unique()
                    z2 = compared_line_xyz[compared_line_xyz["x"] == intersection_x]["z"].unique()

                    if len(z1) == 0 or len(z2) == 0:
                        continue

                    if np.array_equal(z1, z2):  # Endpoint contact
                        if angle_calculator(current_line_linestring, compared_line_linestring) < 30:
                            x1 = ((list(current_line_linestring.coords)[:21])[-3])[0]
                            x2 = ((list(compared_line_linestring.coords)[:21])[-3])[0]

                            z1 = current_line_xyz[current_line_xyz["x"] == x1]["z"].unique()
                            z2 = compared_line_xyz[compared_line_xyz["x"] == x2]["z"].unique()

                            if len(z1) and len(z2) and z1[0] < z2[0]:
                                valid_lines.append(line_id_test)
                        else:
                            valid_lines.append(line_id_test)
                    else:
                        if z2[0] > z1[0]:
                            valid_lines.append(line_id_test)
                elif intersection.geom_type == "MultiPoint":
                    continue  # More complex intersections could be handled if needed
            else:
                if current_line_linestring.distance(compared_line_linestring) > 0.5:
                    valid_lines.append(line_id_test)

    return list(set(valid_lines))  # Remove duplicates

def eulerficator(df, terminal_points, nodes):
    terminal_points_nogroups = terminal_points.reset_index()
    clusters = terminal_points_nogroups['cluster'].unique()
    lines = terminal_points_nogroups['line_id'].unique()

    cluster_dict = {}
    for c in clusters:
        cluster_dict[c] = list(terminal_points_nogroups[terminal_points_nogroups['cluster'] == c]['line_id'].values)

    line_order = []
    line_order_grouped = []
    node_order = []
    node_order_grouped = []
    
    min_z = df.groupby('line_id')['z'].min()
    heightsorted_line_ids = min_z.sort_values().index.tolist()
    unprinted_lines = heightsorted_line_ids.copy()

    while len(line_order) < len(lines):
        valid_lines = validator(unprinted_lines.copy(), df)

        while valid_lines:
            current_line_lines = []
            current_line_nodes = []

            next_line = valid_lines[0]
            line_order.append(next_line)
            current_line_lines.append(next_line)
            unprinted_lines.remove(next_line)
            valid_lines.remove(next_line)

            connected_nodes = terminal_points.loc[next_line]['cluster'].values
            start_node = nodes.loc[connected_nodes]['z'].idxmin()
            end_node = connected_nodes[connected_nodes != start_node][0]

            current_line_nodes.extend([start_node, end_node])
            node_order.extend([start_node, end_node])

            walking = True
            while walking:
                connected_lines = [l for l in cluster_dict[end_node] if l in valid_lines and l in unprinted_lines]

                if not connected_lines:
                    walking = False
                    break

                next_line = connected_lines[0]
                line_order.append(next_line)
                current_line_lines.append(next_line)
                unprinted_lines.remove(next_line)
                valid_lines.remove(next_line)

                connected_nodes = terminal_points.loc[next_line]['cluster'].values
                start_node = end_node
                end_node = connected_nodes[connected_nodes != start_node][0]

                current_line_nodes.append(end_node)
                node_order.append(end_node)

            line_order_grouped.append(current_line_lines)
            node_order_grouped.append(current_line_nodes)

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

    for n in np.unique(df['line_id'].values):
        ax.plot(df[df['line_id'] == n]['x'], df[df['line_id'] == n]['y'], df[df['line_id'] == n]['z'])

    # If you need a specific line plotting
    # n = 2
    # ax.plot(df[df['line_id'] == n]['x'], df[df['line_id'] == n]['y'], df[df['line_id'] == n]['z']

    for n in np.unique(terminal_points['cluster'].values):
        ax.scatter(terminal_points[terminal_points['cluster'] == n]['x'],
                   terminal_points[terminal_points['cluster'] == n]['y'],
                   terminal_points[terminal_points['cluster'] == n]['z'])

    ax.view_init(elev=30, azim=60)  # Elevation of 30 degrees, azimuth of 45 degrees
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
filedir = './test_files/'  # Directory of the file you're loading
# Name of the output file
filename ='3Dblood27.11.txt'   # Name of the file you're loading
fileout = 'file.gcode'

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
eulerficator(df=node_finder(df)[0], terminal_points=node_finder(df)[1], nodes=node_finder(df)[2])
