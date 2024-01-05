import os
from lxml import etree as ET
import networkx as nx
import tqdm

from tqdm import trange

def create_dynamic_gexf(edge_index, dataset, output_file, input_file=None, max_T=2000):
    """
    Function that creates a dynamic GEXF file from a dataset.
    Documentation for adding dynamic attributes:
    https://docs.gephi.org/User_Manual/Import_Dynamic_Data/
    """

    # Create the file if it doesn't exist
    if input_file is None or not os.path.exists(input_file):
        # Create a new graph
        graph = nx.Graph()

        # Add edges from the dataset's connectivity
        graph.add_edges_from(edge_index.T)

        # Export the graph to GEFX format
        nx.write_gexf(graph, input_file, version="1.2draft")

    # Load the GEXF file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find the graph element
    graph_element = root.find("{http://www.gexf.net/1.2draft}graph")
    # Update the mode attribute value
    graph_element.set("mode", "dynamic")
    graph_element.set("timerepresentation", "interval")
    graph_element.set("timeformat", "integer")

    # Add time varying attribute
    attributes = ET.Element("{http://www.gexf.net/1.2draft}attributes", {"class": "node", "mode": "dynamic"})
    _ = ET.SubElement(attributes, "{http://www.gexf.net/1.2draft}attribute", {"id": "pv", "title": "PV", "type": "float"})
    graph_element.insert(0, attributes) # Set the attributes element before nodes

    # Find the nodes element
    nodes = graph_element.find("{http://www.gexf.net/1.2draft}nodes")

    # Loop through all the nodes
    with tqdm.tqdm(total=len(nodes.findall("{http://www.gexf.net/1.2draft}node"))) as pbar:
        for node in nodes.iter("{http://www.gexf.net/1.2draft}node"):
            # Get the id
            node_id = int(node.attrib['id'])

            # Add attribute and time series
            time_series = dataset.target.to_numpy()[:max_T, node_id]
            attvalues = ET.SubElement(node, "{http://www.gexf.net/1.2draft}attvalues")

            # Loop through the time series
            for i in trange(len(time_series), leave=False, desc=f"Node {node_id}"):
                attvalue = ET.SubElement(attvalues, "{http://www.gexf.net/1.2draft}attvalue",
                                        {"for": "pv", "value": str(time_series[i]), "start": str(i*10), "end": str((i+1)*10)})

            # Update the progress bar
            pbar.update(1)

    # Save the modified GEXF file
    tree.write(output_file, xml_declaration=True, encoding="utf-8")