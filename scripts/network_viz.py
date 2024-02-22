import pandas as pd
import numpy as np
import os
import pdb
import warnings
from graph_tool.all import *
import seaborn as sns
from config import *
from src.data_processing import merge_data

X_percent = 1.0
COLOR_START = 0
COLOR_END = 0.75
WHICH_INFLUENCE = 'single' # potential, real, or single
WHICH_GROUP = 'both'
EGONETS = False

def get_knn_influence(distance_matrix):
    """
    Get knn influence for k = 5 for a given distance matrix

    Args:
        distance_matrix (np array): shape num_raters x num_raters

    Returns:
        nearest_matrix (np array): top 5

    """
    nearest_matrix = np.zeros((1, 3))
    for row in range(distance_matrix.shape[0]):
        rater_row = np.copy(distance_matrix[row, :])
        rater_row[np.isnan(rater_row)] = -1        
        rater_row[row] = 0 # self distance = 0
        rater_row[rater_row < np.sort(rater_row)[-5]] = 0  # all those values less than the 5th greatest value
        rater_row = rater_row / np.sum(rater_row)  # top k influences, summing to 1.0
        from_column = [row for i in range(np.sum(rater_row > 0))]
        to_column = np.where(rater_row > 0)[0]
        value_column = rater_row[rater_row > 0]
        nearest_matrix = np.concatenate([nearest_matrix, np.array([from_column, to_column, value_column]).transpose()])
    nearest_matrix = nearest_matrix[1:,] # take off the top row
    return nearest_matrix

if WHICH_INFLUENCE == 'potential':
    master_dataset_c = merge_data()[0]
    potential_corr = pd.DataFrame(master_dataset_c.T).corr(min_periods=5)
    potential_influence_corr = get_knn_influence(potential_corr.to_numpy())
    all_adjacencies = pd.DataFrame(potential_influence_corr, columns=['from_node','to_node','weight'])
elif WHICH_INFLUENCE == 'real':
    chosen_parameter = PARAMETERS['k'].index(5) * len(PARAMETERS['rho']) + PARAMETERS['rho'].index(1)
    real_adjacencies = pd.read_csv(os.path.join('results', 'simulations',WHICH_GROUP,'adjacencies',
                                                str(chosen_parameter)+'.csv'),header=None, sep=' ')
    real_adjacencies.columns = ['to', 'from', 'value']
    all_adjacencies = real_adjacencies
    all_adjacencies.columns = ['from_node', 'to_node', 'weight']
elif WHICH_INFLUENCE == 'single':
    real_adjacencies = pd.read_csv(os.path.join('results', 'simulations', 'supplementary', 'single_wines', 'adjacencies_1860.csv'),header=None, sep=' ')
    real_adjacencies.columns = ['to', 'from', 'value']
    all_adjacencies = real_adjacencies
    all_adjacencies.columns = ['from_node', 'to_node', 'weight']
    

# read rmses and drop nans
all_rmses = np.array(pd.read_csv(os.path.join('results','simulations',WHICH_GROUP,'performance.csv'), 
                                 header=None, sep=' '))
error_index = PARAMETERS['k'].index(5) * len(PARAMETERS['rho']) + PARAMETERS['rho'].index(1)
all_rmses = pd.DataFrame(all_rmses[:,error_index])

unique_nodes = all_adjacencies['from_node'].unique()
pareto_nw = pd.DataFrame(columns=['from_node','to_node','weight'])
for node in unique_nodes:
    subset_data = all_adjacencies.loc[all_adjacencies['from_node'] == node,:]
    subset_data = subset_data.sort_values(by='weight', ascending=False)
    if X_percent < 1: 
        val_to_reach = sum(subset_data['weight'])  * X_percent
        subset_data = subset_data.loc[subset_data['weight'].cumsum() < val_to_reach,:]
    #subset_data = subset_data.iloc[0:12,:] # choose top N only
    subset_data['weight'] = (subset_data['weight']/sum(subset_data['weight']))*2
    pareto_nw = pareto_nw.append(subset_data)

ex_am_labels = ['e' for e in range(NUM_EXPERTS)] + ['a' for a in range(NUM_AMATEURS)]
# Set colours
flare_colors = sns.color_palette("flare",n_colors=100)
colour_map = {}
colour_map['a'] = tuple(flare_colors[int(COLOR_START * len(flare_colors))])
colour_map['e'] = tuple(flare_colors[int(COLOR_END * len(flare_colors))])
colour_map['a'] = colour_map['a'] + (0.25,)
colour_map['e'] = colour_map['e'] + (0.25,)


full_network = Graph()
edge_weights = full_network.new_edge_property('double')
edge_colour = full_network.new_edge_property('vector<double>')
for ind,node in enumerate(unique_nodes):
    subset_data = pareto_nw.loc[pareto_nw['from_node']==node,:]
    for row in subset_data.index:
        e = full_network.add_edge(int(subset_data.loc[row,'from_node']),int(subset_data.loc[row,'to_node']))
        if EGONETS:
            edge_weights[e] = subset_data.loc[row,'weight'] * 14
        else:
            edge_weights[e] = subset_data.loc[row, 'weight'] * 7
        #to_label = int(subset_data.loc[row,'from_node'])
        #edge_colour[e] = colour_map[ex_am_labels[list(unique_nodes).index(to_label)]]
        to_label = int(subset_data.loc[row,'to_node'])
        edge_colour[e] = colour_map[ex_am_labels[list(unique_nodes).index(to_label)]]

deg = full_network.degree_property_map("in")

all_vertices = full_network.get_vertices()
# add additional vertex properties
vertex_labels = full_network.new_vertex_property('string')
node_rmse = full_network.new_vertex_property('vector<double>')
vertex_colour = full_network.new_vertex_property('vector<double>')
node_size = full_network.new_vertex_property('double')
# node size is the weights supplied/ max, ie., node size = sum(to_node==node[weight])
node_contribution = np.zeros(all_vertices.shape)
for ind,ver in enumerate(all_vertices):
    temp_subset = pareto_nw.loc[pareto_nw['to_node']==ind,:]
    node_contribution[ind] = sum(temp_subset['weight'])# Don't sum just yet    /sum(pareto_nw['weight'])

node_contribution = node_contribution + np.min(node_contribution[node_contribution>0])
node_contribution = node_contribution/np.sum(node_contribution)
#np.savetxt(Path(Path.cwd()).parents[0].parents[0].joinpath('Data/vis/data/actual_influence.csv'), node_contribution,
#                                                            delimiter=",")

expert_labels = {'WA':'Lisa Perotti Brown','NM':'Neal Martin','JR':'Jancis Robinson',
                  'TA':'Tim Atkin','B&D':'Bettane and Desseauve','JS':'James Suckling',
                   'JL':'Jeff Leve','De':'Jane Anson','RVF':'Poels, Durand and Maurange',
                   'JA':'Jane Anson','LeP':'Dupont','PW':'de Groot','RG':'Rene Gabriel','CK':'Chris Kissack'}
expert_label_keys = list(expert_labels.keys())

# make sure the rmse colours lie in 0-1 range instead of the current range for example - 0.829 - 0.52
max_rmse, min_rmse = np.max(all_rmses.iloc[:,0]), np.min(all_rmses.iloc[:,0])
print(max_rmse, min_rmse)
#rmse_colours = (all_rmses.iloc[:,0] - min_rmse)/(max_rmse - min_rmse) ###### CHANGE HERE FOR OPPOSITE FILL COLOURS
rmse_colours = (max_rmse - all_rmses.iloc[:,0])/(max_rmse - min_rmse)

for ind,ver in enumerate(all_vertices):
    node_size[full_network.vertex(ver)] = (node_contribution[ind]**0.4) * 100#((node_contribution[ind]) ** 0.4) * 100
    node_rmse[full_network.vertex(ver)] = tuple([rmse_colours.iloc[ind], rmse_colours.iloc[ind], rmse_colours.iloc[ind], .5])
    vertex_colour[full_network.vertex(ver)] = colour_map[ex_am_labels[ind]]
    if ind < 14:
        vertex_labels[full_network.vertex(ver)] = expert_label_keys[ind]
    else:
        vertex_labels[full_network.vertex(ver)] = ''

full_network.vertex_properties['vertex_colour'] = vertex_colour
full_network.vertex_properties['node_rmse'] = node_rmse
full_network.vertex_properties['vertex_labels'] = vertex_labels
full_network.vertex_properties['node_size'] = node_size
full_network.edge_properties['edge_colour'] = edge_colour
full_network.edge_properties['edge_thickness'] = edge_weights

pos = radial_tree_layout(full_network, root=0, weighted=False,
                        node_weight=full_network.vertex_properties['node_size'])


sum_node_sizes = 0
all_node_sizes = []
for ind, vert in enumerate(full_network.vertices()):
    all_node_sizes.append(full_network.vertex_properties['node_size'][ind])
    
######Arrange nodes in order of size within experts, within amateurs
expert_node_sizes = all_node_sizes[: NUM_EXPERTS].copy()
amateur_node_sizes = all_node_sizes[NUM_EXPERTS : ].copy()
sorted_indices = np.concatenate([np.argsort(expert_node_sizes), NUM_EXPERTS + np.argsort(amateur_node_sizes)])
sorted_indices = sorted_indices.tolist()

# now get the angle offset for all nodes :
sum_node_sizes = np.sum(all_node_sizes)
angle_cuts = []
for ind in sorted_indices:
    angle_cut = (all_node_sizes[ind]/sum_node_sizes) * 360
    angle_cuts.append(angle_cut)
all_node_sizes = [all_node_sizes[i] for i in sorted_indices]
# set x, y values for all nodes.
#RADIUS = 1
vert_objs = list(full_network.vertices())
vert_objs = [vert_objs[i] for i in sorted_indices]
#for ind,vert in enumerate(vert_objs):
#    print(vert_objs[])
#pdb.set_trace()
start_angle = 0.0
vertex_text_pos = full_network.new_vertex_property('float')
vertex_font_size = full_network.new_vertex_property('int')
vertex_text_off = full_network.new_vertex_property('vector<double>')
for ind,vert in enumerate(vert_objs):
    current_pointer = start_angle + (angle_cuts[ind]/2)
    start_angle = current_pointer + (angle_cuts[ind]/2)
    new_pos = (np.cos(np.deg2rad(current_pointer)), np.sin(np.deg2rad(current_pointer)))
    vertex_text_pos[vert] = -2#"#np.deg2rad(current_pointer)
    new_pos = np.array(new_pos)
    try:
        pos[vert] = new_pos
    except:
        pdb.set_trace()
    vertex_text_off[vert] = [new_pos[0] * .02 * np.sqrt(all_node_sizes[ind]),
                             new_pos[1] * .02 * np.sqrt(all_node_sizes[ind])]
    vertex_font_size[vert] = 40
full_network.vertex_properties['text_position'] = vertex_text_pos
full_network.vertex_properties['text_offset'] = vertex_text_off
full_network.vertex_properties['font_size'] = vertex_font_size
    
state = minimize_nested_blockmodel_dl(full_network)
t = get_hierarchy_tree(state)[0]
tpos = pos
#tpos = pos = radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
cts = get_hierarchy_control_points(full_network, t, tpos)
if (WHICH_INFLUENCE == 'potential') or (WHICH_INFLUENCE == 'real'):
    graph_draw(full_network, pos, fmt='png',
           output=os.path.join('results','visualization', 'fig_3b.png'),
           vertex_color=full_network.vertex_properties['vertex_colour'], # only circumference of vertex
           vertex_fill_color=full_network.vertex_properties['node_rmse'],
           vertex_size=full_network.vertex_properties['node_size'],
           edge_color=full_network.edge_properties['edge_colour'],
           edge_pen_width=full_network.edge_properties['edge_thickness'],
           penwidth=1.5, halo=True, 
           halo_color=full_network.vertex_properties['vertex_colour'], 
           ink_scale=1.5, bg_color='white', 
           fit_view=True, output_size=(1200,1200),
           edge_control_points=cts,
           anchor=1, ratio='auto',
           font_size=full_network.vertex_properties['font_size'], 
           vertex_text=full_network.vertex_properties['vertex_labels'],
           vertex_text_offset=full_network.vertex_properties['text_offset'],
           vertex_text_color='black',
           vertex_text_postion=full_network.vertex_properties['text_position'])
else:
    graph_draw(full_network, pos, fmt='png',
           output=os.path.join('results','visualization', 'fig_7d.png'),
           vertex_color=full_network.vertex_properties['vertex_colour'], # only circumference of vertex
           vertex_fill_color=full_network.vertex_properties['node_rmse'],
           vertex_size=full_network.vertex_properties['node_size'],
           edge_color=full_network.edge_properties['edge_colour'],
           edge_pen_width=full_network.edge_properties['edge_thickness'],
           penwidth=1.5, halo=True, 
           halo_color=full_network.vertex_properties['vertex_colour'], 
           ink_scale=1.5, bg_color='white', 
           fit_view=True, output_size=(1200,1200),
           edge_control_points=cts,
           anchor=1, ratio='auto')



















