from operator import mul
import networkx as nx
import os
import sys
import community
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
import numpy as np

#                            class Communautes                                 
class Communities:
    def __init__(self, ipt_txt, ipt_png):
        self.ipt_txt = ipt_txt
        self.ipt_png = ipt_png
        self.graph = None

    def initialize(self):
        if not os.path.isfile(self.ipt_txt):
            self.quit(self.ipt_txt + " doesn't exist or it's not a file")
        # initialiser les bibliotheques 
        self.graph = nx.Graph()
        # lire les donnees
        self.load_txt(self.ipt_txt

    #                              Fonctions principales                         
    def find_best_partition(self):
        G = self.graph.copy()
        modularity = 0.0
        removed_edges = []
        partition = {}
        while 1:
            betweenness = self.calculte_betweenness(G)
            max_betweenness_edges = self.get_max_betweenness_edges(betweenness)
            if len(G.edges()) == len(max_betweenness_edges):
                break

            G.remove_edges_from(max_betweenness_edges)  
            components = nx.connected_components(G)
            idx = 0
            tmp_partition = {}
            for component in components:
                for inner in list(component):
                    tmp_partition.setdefault(inner, idx)
                idx += 1
            cur_mod = community.modularity(tmp_partition, G)

            if cur_mod < modularity:
                G.add_edges_from(max_betweenness_edges)
                break;
            else:
                partition = tmp_partition
            removed_edges.extend(max_betweenness_edges)
            modularity = cur_mod
        return partition, G, removed_edges

    def get_max_betweenness_edges(self, betweenness):
        max_betweenness_edges = []
        max_betweenness = max(betweenness.items(), key=lambda x: x[1])
        for (k, v) in betweenness.items():
            if v == max_betweenness[1]:
                max_betweenness_edges.append(k)
        return max_betweenness_edges

    def calculte_betweenness(self, G, bonus=True):
       #Calculer Betweenness 
		"""        entree: 
		- G: graphique 
		- Bonus: True si utiliser ma propre calculatrice betweenness. (Bonus = True par defaut) 
        """
        if bonus:
            betweenness = self.my_betweenness_calculation(G)
        else:
            betweenness = nx.edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None)
        return betweenness

    #                               Fonctions auxiliaire                              
    def build_level(self, G, root):
        """ 
        Niveau pour le graphe
		entree: 
		- G: networkx graph 
		- racine : noeud racine 
		sortie : 
		- niveaux: noeuds dans chaque niveau 
		- predecesseurs : predecesseurs pour chaque noeud 
		- successeurs : successeurs pour chaque noeud 
        """
        levels = {}
        predecessors = {}
        successors = {}

        cur_level_nodes = [root]    # initialize le point de depart
        nodes = []  # stocke les noeuds qui ont ete consultes
        level_idx = 0       
        while cur_level_nodes:  # si ont des noeuds pour un niveau, continuer le processus
            nodes.extend(cur_level_nodes)  
             # ajouter des noeuds qui sont a l'interieur de nouveau niveau dans la liste des noeuds
            levels.setdefault(level_idx, cur_level_nodes)  
             # definir des noeuds pour niveau actuel
            next_level_nodes = []   # preparer des noeuds pour les niveaux suivants

            # trouver noeud dans la prochaine etape
            for node in cur_level_nodes:
                nei_nodes = G.neighbors(node)   # tous les voisins pour le noeud dans le niveau actuel
                # trouver des noeuds voisins dans le niveau suivant
                for nei_node in nei_nodes:
                    if nei_node not in nodes:   # noeuds dans le niveau suivant ne doit pas etre accede
                        predecessors.setdefault(nei_node, [])   
                        # initialiser le dictionnaire predecesseurs, utiliser une liste pour stocker tous les predecesseurs
                        
                        predecessors[nei_node].append(node) 
                        successors.setdefault(node, [])     
                        # initialiser le dictionnaire successeurs, utiliser une liste pour stocker tous les successeurs
                        
                        successors[node].append(nei_node)

                        if nei_node not in next_level_nodes:    # eviter d'ajouter meme noeud deux fois
                            next_level_nodes.append(nei_node)
            cur_level_nodes = next_level_nodes
            level_idx += 1
        return levels, predecessors, successors

    def calculate_credits(self, G, levels, predecessors, successors, nodes_nsp):
        """
        Calculer les credits pour les noeuds et les bords
        """
        nodes_credit = {}
        edges_credit = {}
        # boucle, de bas en haut, sans inclure le niveau zero
        for lvl_idx in range(len(levels)-1, 0, -1):
            lvl_nodes = levels[lvl_idx]     # obtenir des noeuds dans le niveau
            # calculer pour chaque noeud du niveau actuel
            for lvl_node in lvl_nodes:
                nodes_credit.setdefault(lvl_node, 1.)   
                # definir le credit par defaut pour le noeud, 1
                if successors.has_key(lvl_node):        
                # si ce n'est pas un noeud feuille
                    # Chaque noeud qui n'est pas une feuille obtient le credit = 1 + somme des credits des bords  de ce noeud au niveau ci-dessous
                    for successor in successors[lvl_node]:
                        nodes_credit[lvl_node] += edges_credit[(successor, lvl_node)]
                node_predecessors = predecessors[lvl_node]  
                #  obtenir des predecesseurs du noeud dans le niveau actuel
                
                total_nodes_nsp = .0    
                # nombre total de chemins plus courts pour les predecesseurs du noeud au niveau actuel
                
                for predecessor in node_predecessors:
                    total_nodes_nsp += nodes_nsp[predecessor]

                for predecessor in node_predecessors:
                    predecessor_weight = nodes_nsp[predecessor]/total_nodes_nsp     
                    edges_credit.setdefault((lvl_node, predecessor), nodes_credit[lvl_node]*predecessor_weight)       
        return nodes_credit, edges_credit

    def my_betweenness_calculation(self, G, normalized=False):
    #    Fonction de calcul du betweenness
        graph_nodes = G.nodes()
        edge_contributions = {}
        components = list(nx.connected_components(G))   
        # composants connectes pour graph en cours
        
        # calculer pour chaque noeud
        for node in graph_nodes:
            component = None    # le courant communautaire auquel appartient le noeud
            for com in components: 
                if node in list(com):
                    component = list(com)
            nodes_nsp = {}  # numbere de "shorest paths"
            node_levels, predecessors, successors = self.build_level(G, node)   
            #  construction des niveaux pour les calculs
            
            # calculer des chemins plus courts pour chaque noeud (y compris le noeud actuel)
            for other_node in component:
                shortest_paths = nx.all_shortest_paths(G, source=node,target=other_node)
                nodes_nsp[other_node] = len(list(shortest_paths))
            # calculer des credits pour les noeuds et les bords (Utilisez uniquement "edges_credit" en fait)
            nodes_credit, edges_credit = self.calculate_credits(G, node_levels, predecessors, successors, nodes_nsp)

            # triple de tri (valeur cle de edges_credit), et resumer pour edge_contributions
            for (k, v) in edges_credit.items():
                k = sorted(k, reverse=False)
                edge_contributions_key = (k[0], k[1])
                edge_contributions.setdefault(edge_contributions_key, 0)
                edge_contributions[edge_contributions_key] += v
           
        # diviser par 2 pour obtenir la vraie betweenness
        for (k, v) in edge_contributions.items():
            edge_contributions[k] = v/2

        # Normaliser
        if normalized:
            max_edge_contribution = max(edge_contributions.values())
            for (k, v) in edge_contributions.items():
                edge_contributions[k] = v/max_edge_contribution
        return edge_contributions

#                             Methode Main	                               
if __name__ == '__main__':

    ipt_txt = sys.argv[1]
    ipt_png = sys.argv[2]

    c = Communities(ipt_txt, ipt_png)
    c.initialize()
    partition, part_graph, removed_edges = c.find_best_partition()
    c.display(partition)
    c.plot_graph(part_graph, removed_edges)

