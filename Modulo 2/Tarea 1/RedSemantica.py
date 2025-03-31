import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_nodes_from([
    "Página principal",
    "Servicios",
    "Productos",
    "Blog",
    "SEO",
    "Marketing de contenidos",
    "Herramientas SEO"
])

G.add_edges_from([
    ("Página principal", "Servicios"),
    ("Página principal", "Productos"),
    ("Página principal", "Blog"),
    ("Servicios", "SEO"),
    ("Servicios", "Marketing de contenidos"),
    ("SEO", "Herramientas SEO"),
    ("Blog", "SEO"),
    ("Blog", "Marketing de contenidos")
])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold")
plt.show()