#!/usr/bin/env python3
from pandas import read_csv
from matplotlib import pyplot
import json, sys


def define_figure():
    pyplot.rc('axes', titlesize=22, labelsize=28)
    pyplot.rc('xtick', labelsize=24)
    pyplot.rc('ytick', labelsize=24)
    pyplot.rc('legend', fontsize=18)
    return pyplot.subplots(dpi=200, figsize=(11, 7))

def graph(genes_to_graph:list, title:str, graph_types:list):
    data_df = read_csv("similarities.csv", "r")
    genes_to_graph = genes_to_graph or data_df["gene1"]+data_df["gene2"]
    data_to_graph = data_df[data_df["gene1"].isin(genes_to_graph) | data_df["gene2"].isin(genes_to_graph)]
    if "seq_str" or "str_seq" in graph_types:
        # plot sequence v structure
        genes_w_structure = data_to_graph[~data_to_graph["structure"].isnull()]
        to_graph = genes_w_structure[~genes_w_structure["sequence"].isnull()]
        xs, ys = to_graph["structure"], to_graph["sequence"]
        x_label, y_label = "structure", "sequence"
        if "seq_str" in graph_types:
            xs, ys = to_graph["sequence"], to_graph["structure"]
            x_label, y_label = "sequence", "structure"
        fig, ax = define_figure()
        ax.scatter(xs, ys)
        ax.set_xlabel(x_label) ; ax.set_ylabel(y_label) ; ax.grid() ; ax.set_title(title)
    elif "seq_fun" or "fun_seq" in graph_types:
        # plot sequence v function
        genes_w_functions = data_to_graph[~data_to_graph["function"].isnull()]
        to_graph = genes_w_functions[~genes_w_functions["sequence"].isnull()]
        xs, ys = to_graph["function"], to_graph["sequence"]
        x_label, y_label = "function", "sequence"
        if "seq_fun" in graph_types:
            xs, ys = to_graph["sequence"], to_graph["function"]
            x_label, y_label = "sequence", "function"
        fig, ax = define_figure()
        ax.scatter(xs, ys)
        ax.set_xlabel(x_label) ; ax.set_ylabel(y_label) ; ax.grid() ; ax.set_title(title)
    elif "fun_str" or "str_fun" in graph_types:
        # plot structure v function
        genes_w_functions = data_to_graph[~data_to_graph["function"].isnull()]
        to_graph = genes_w_functions[~genes_w_functions["structure"].isnull()]
        xs, ys = to_graph["structure"], to_graph["function"]
        x_label, y_label = "structure", "function"
        if "fun_str" in graph_types:
            xs, ys = to_graph["function"], to_graph["structure"]
            x_label, y_label = "function", "structure"
        fig, ax = define_figure()
        ax.scatter(xs, ys)
        ax.set_xlabel(x_label) ; ax.set_ylabel(y_label) ; ax.grid() ; ax.set_title(title)

if sys.argv[1] == "all":      # plots all the data
    import pickle
    # load all of the data
    with open("all_genes_list.txt", "rb") as all_genes:
        data = pickle.load(all_genes)
    title = "All genes"

elif "EC:" in sys.argv[1]:    # plots all data associated with EC number
    with open("EC_genes.json", "r") as ec:
        ec_genes = json.load(ec)
    ecNum = sys.argv[1].split(":")[1]
    data = ec_genes[ecNum]
    title = f"Genes that correspond with an EC of {ecNum}"

elif "MS:" in sys.argv[1]:    # plots all for rxn00001
    with open("RXN_genes.json", "r") as rxn:
        rxn_genes = json.load(rxn)
    rxnID = sys.argv[1].split(":")[1]
    data = rxn_genes[rxnID]
    title = f"Genes that correspond with the {rxnID} reaction"

elif "SEED:" in sys.argv[1]:  # plots all associated with specified SEED role
    with open("role_genes.json", "r") as role:
        role_genes = json.load(role)
    role = sys.argv[1].split(":")[1]
    data = role_genes[role]
    title = f"Genes that correspond with the {role} role"

elif "G:" in sys.argv[1]:     # plots all for a genome
    geneID = sys.argv[1].split(":")[1]
    data = [geneID]
    title = f"The {geneID} gene"

elif "FAM:" in sys.argv[1]:   # plots all for a family
    with open("fam_genes.json", "r") as fam:
        fam_genes = json.load(fam)
    famID = sys.argv[1].split(":")[1]
    data = fam_genes[famID]
    title = f"The {famID} family of genes"

else:
    raise ValueError(f"The first provided argument ({sys.argv[1]}) is not accepted.")


# graph the aforedefined criteria
graph_types = sys.argv[2] or ["seq_str", "seq_fun", "fun_str"]
graph(data, title, graph_types)
