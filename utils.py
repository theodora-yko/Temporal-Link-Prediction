import matplotlib.pyplot as plt
from functools import reduce
from IPython.core.display import display, HTML
import networkx as nx

# helper functions
def add_nodes_given_df(graph, df, columns): 
    """ 
    Add nodes from a dataframe to the graph given list of columns
    note: the capacity (weight in networkx) of the edge is the total load count between the two locations,
    must have the column name total_loads
    """
    for column in columns:
        graph.add_nodes_from(df[column].unique())
    return graph

def add_edges_given_graph(network_graph, an_inbound_df, an_outbound_df, location_grouping='kma'): 
    """
    NOTE) must be differentiated from networkX library's add_edges() function
    """
    for _, row in an_inbound_df.iterrows(): 
        # origin -> facility 1 
        network_graph.add_edge(row[f'origin_{location_grouping}_id'], row.facility_id, \
                               time = int(row.load_date.strftime("%Y%m%d")))
        #, capacity = row.total_loads)

    for _, row in an_outbound_df.iterrows():
         # facility 1 -> facility 2
        network_graph.add_edge(row.facility_id, row[f'destination_{location_grouping}_id'], \
            time = int(row.load_date.strftime("%Y%m%d")))#, capacity = row.total_loads)
        # facility 2 -> origin 
        network_graph.add_edge(row[f'facility_{location_grouping}_id'], \
            row[f'destination_{location_grouping}_id'], \
            time = int(row.load_date.strftime("%Y%m%d")))#, capacity = row.total_loads) 
    return network_graph

def find_top_keys(test_dict, top_n=10, make_list = True): 
    """
    ** Helper function **
    Given a dictionary, 
    returns: sort the dictionary by value and return the top n keys (list)
    """
    sorted_tuple = sorted(test_dict.items(), key=lambda x:x[1])[::-1]
    if not make_list: return sorted_tuple[:top_n]
    return [x[0] for x in sorted_tuple[:top_n]]

def flatten_tuple(nested_tuple):
    def reducer(acc, val):
        if isinstance(val, tuple):
            return acc + flatten_tuple(val)
        else:
            return acc + (val,)
 
    return reduce(reducer, nested_tuple, ())

def find_max_capacity(edge_weights_dict):
    max_capacity = 0
    for i in edge_weights_dict.values(): 
        if max_capacity < i['capacity']: max_capacity = i['capacity']
    return max_capacity

def plt_helper(fig=None, function="create"):
    """
    helper function for visualization purposes 
    function = "create" or "show"

    returns: either a fig, ax or shows the fig
    """
    if function == 'create': 
        new_fig, new_ax = plt.subplots(figsize=(18, 18))
        return new_fig, new_ax
    elif function == "show": 
        fig.show()

def filter_given_keys(dictionary, keys, return_type=dict): 
    """
    helper function
    """
    if return_type == 'dict': 
        new_dict = {} 
        for key in keys: 
            new_dict[key] = dictionary[key]
        return new_dict
    elif return_type == "list": 
        new_list = [] 
        for key in keys: 
            new_list.append(list(dictionary[key]))
        return new_list

## dataframes display helper function 
def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day