import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys


def is_valid_extension(file_path):
    # Check if the extension is jpeg
    valid_extension = ('.jpg', '.jpeg', '.JPG', '.JPEG')
    _, extension = os.path.splitext(file_path)

    if extension not in valid_extension:
        return False

    return True


def add_graph_data(parent_path, data):
    parent_name = parent_path.split("/")[-1]

    # Count number of valid image in the current Directory
    plants_nbr = sum(1 for entry in os.scandir(parent_path)
                     if entry.is_file() and is_valid_extension(entry.path))

    # Add it to the graph if there is at least 1 valid image
    if plants_nbr >= 1:
        print(f"Dir: {parent_path}: Added to Data Graph")
        data["categories"].append(parent_name)
        data["nbr"].append(plants_nbr)

    child_entries = os.scandir(parent_path)

    # Recursive call on Sub-Directories
    for elem in child_entries:
        child_path = parent_path + "/" + elem.name
        if os.path.isdir(child_path):
            if not os.access(child_path, os.R_OK):
                print(f"Error: Can't access: {child_path}"
                      ", Read access Denied.")
                print("-> Data not taken into account")
            else:
                add_graph_data(child_path, data)


def create_graphs(dir_name, data):
    if not data["categories"] or not data["nbr"]:
        print("No valid File found, Graph can't be drawn")
        exit(0)

    default_colors = plotly.colors.qualitative.Plotly

    # Setup the Graph Area
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )

    # Create the Pie Graph
    fig.add_trace(
        go.Pie(labels=data['categories'], values=data['nbr'],
               marker=dict(colors=default_colors)),
        row=1, col=1
    )

    # Create the Bar Graph
    fig.add_trace(
        go.Bar(x=data['categories'], y=data['nbr'],
               marker_color=default_colors),
        row=1, col=2
    )

    # Set Graph Title
    title = dir_name + " class distribution"
    fig.update_layout(title=title)
    fig.show()


if __name__ == "__main__":

    # Checks Inputs Validity
    if len(sys.argv) <= 1 or len(sys.argv) > 2:
        print("Error: Expected Argument: 'Path to Directory'")
        exit(1)

    input_path = sys.argv[1]
    input_dir_name = input_path.split("/")[-1]

    if not os.path.exists(input_path) or not os.path.isdir(input_path):
        print(f"Error: Invalid Input Path: {input_path}")
        exit(1)
    if not os.access(input_path, os.R_OK):
        print(f"Error: Can't access: {input_path}, Read access Denied")
        exit(1)

    # Data Dictionary: add data categories, count valid images/category
    data = {
        "categories": [],
        "nbr": []
    }

    add_graph_data(input_path, data)

    create_graphs(input_dir_name, data)
