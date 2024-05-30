import os
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from PIL import Image
from dataset.utils import load_classification_dataset


def make_gif(dataset_name, graph_idx):
    # Load dataset
    edge_indexes, node_labels, graph_labels = load_classification_dataset(dataset_name, False)

    all_edges = []
    for e in edge_indexes[graph_idx]:
        all_edges.append(e.T)
    all_edges = np.concatenate(all_edges, axis=0)
    node_offset = all_edges.min()
    all_edges = np.unique(all_edges, axis=0) - node_offset

    # Specify the folder path to save the images
    folder_path = "./plots/graphs"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a list to store the images
    images = []

    # Iterate over the edge_indexes list: t=i, edge_index is the connectivity at time t
    for i, edge_index in enumerate(edge_indexes[graph_idx]):

        # Get the edges and the nodes for the current timestep
        edges = edge_index.T
        index_edges = []
        index_nodes = []
        for e in edges:
            e = e - node_offset
            rem_dupl = torch.sort(torch.tensor(all_edges), dim=1).values.unique(dim=0)
            test = (rem_dupl[:,0] == e[0]) * (rem_dupl[:,1] == e[1])
            if test.nonzero().numel():
                index_edges.append(test.nonzero().squeeze())
            index_nodes.append(e[0].item())
            index_nodes.append(e[1].item())

        # Create a new graph
        graph = nx.Graph()

        # Add edges to the graph
        graph.add_edges_from(all_edges.tolist())

        # Set colors and styles for edges
        edge_colors = np.array(["tab:blue" for i in range(len(all_edges))])
        edge_colors[index_edges] = 'tab:red'
        widths = np.array([1 for i in range(len(all_edges))])
        widths[index_edges] = 5
        styles = np.array(['dashed' for i in range(len(all_edges))])
        styles[index_edges] = 'solid'

        # Set colors and style for nodes
        node_colors = np.array(["tab:blue" for i in range(all_edges.max()+1)], dtype='<U9')
        node_colors[index_nodes] = 'tab:red'
        # Change infected nodes' style and size
        node_size = np.array([200 for i in range(all_edges.max()+1)])
        node_size[node_labels[graph_idx][i].squeeze().to(torch.bool)] = 400
        node_colors[node_labels[graph_idx][i].squeeze().to(torch.bool)] = 'tab:green'

        # Plot the graph
        plt.figure(figsize=(7, 7))
        pos = nx.kamada_kawai_layout(graph)
        nx.draw(graph, pos=pos, node_color=node_colors, node_size=node_size, edge_color=edge_colors, style=styles, width=widths, with_labels=False)

        # Add time-step and label in the background
        plt.text(-0.5, 0, f"Timestep {i}", fontsize=20, ha='center', va='bottom', alpha=0.5)
        plt.text(+0.5, 0, f"Label {int(graph_labels[graph_idx])}", fontsize=20, ha='center', va='bottom', alpha=0.5)

        # Save the plot as an image
        plt.savefig(os.path.join(folder_path, f"graph_{i+1}.png"))

        # Close the plot to free up memory
        plt.close()

        # Open the saved image and append it to the list
        image = Image.open(os.path.join(folder_path, f"graph_{i+1}.png"))
        images.append(image)

    # Save the list of images as a GIF
    images[0].save(os.path.join(folder_path, f"graph{graph_idx}_animation.gif"),
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0)

    for i in range(len(images)): os.remove(os.path.join(folder_path, f"graph_{i+1}.png"))


if __name__ == "__main__":
    # Launch example:
    # python -m utils.animation_gif "facebook_ct1" 2

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create GIF animation of graph visualization")

    # Add the command line arguments
    parser.add_argument("dataset_name", type=str, help="Name of the dataset", default="facebook_ct1")
    parser.add_argument("graph_idx", type=int, help="Index of the graph", default=0)

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main_gif function with the parsed arguments
    make_gif(args.dataset_name, args.graph_idx)