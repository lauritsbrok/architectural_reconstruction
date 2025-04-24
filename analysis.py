import os
import re
from git import Repo
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

ZEEGUU_WEB_FOLDER = "./Zeeguu-Web"

def get_top_level_folder(rel_path):
    # Most JS files will be in src/something
    parts = rel_path.split(os.sep)
    if len(parts) > 1 and parts[0] == 'src':
        # Return the second level folder as the module
        return parts[1] if len(parts) > 2 else 'root'
    return parts[0] if len(parts) > 1 else 'root'

def find_source_files():
    """Find all JavaScript files in the Zeeguu-Web folder"""
    source_files = []
    for dirpath, _, filenames in os.walk(ZEEGUU_WEB_FOLDER):
        for filename in filenames:
            if filename.endswith(".js"):
                source_files.append(os.path.join(dirpath, filename))
    return source_files

def extract_imports(file_path):
    """Extract all local (relative) imports from a JavaScript file"""
    imports = set()
    import_pattern = re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]|require\(['\"](.*?)['\"]\)")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            matches = import_pattern.findall(line)
            for match in matches:
                import_path = match[0] or match[1]
                # Only consider relative imports (starting with .)
                if import_path and import_path.startswith('.'):
                    imports.add(import_path)
    return list(imports)

def resolve_relative_path(import_path, from_file):
    """Resolve a relative import path to an absolute file path"""
    from_dir = os.path.dirname(from_file)
    abs_path = os.path.normpath(os.path.join(from_dir, import_path))
    
    # Try different extensions and index.js pattern
    if not os.path.splitext(abs_path)[1]:
        # Try with .js extension
        if os.path.exists(abs_path + '.js'):
            return os.path.relpath(abs_path + '.js', ZEEGUU_WEB_FOLDER)
        # Try index.js in the directory
        if os.path.exists(os.path.join(abs_path, 'index.js')):
            return os.path.relpath(os.path.join(abs_path, 'index.js'), ZEEGUU_WEB_FOLDER)
    elif os.path.exists(abs_path):
        return os.path.relpath(abs_path, ZEEGUU_WEB_FOLDER)
    
    return None

print(f"Finding source files in {ZEEGUU_WEB_FOLDER}...")
source_files = find_source_files()
print(f"Found {len(source_files)} JavaScript files")

module_edges = Counter()
found_imports = 0
resolved_imports = 0

for file_path in source_files:
    rel_path = os.path.relpath(file_path, ZEEGUU_WEB_FOLDER)
    src_module = get_top_level_folder(rel_path)
    imports = extract_imports(file_path)
    
    if imports:
        found_imports += len(imports)
    
    for imp in imports:
        resolved = resolve_relative_path(imp, file_path)
        if resolved:
            resolved_imports += 1
            dst_module = get_top_level_folder(resolved)
            if src_module != dst_module:
                module_edges[(src_module, dst_module)] += 1
                if len(module_edges) < 10:  # Print first few edges for debugging
                    print(f"Added edge: {src_module} -> {dst_module}")

print(f"Found {found_imports} total local imports")
print(f"Successfully resolved {resolved_imports} imports")
print(f"Found {len(module_edges)} unique module dependencies")

# Build the graph
G = nx.DiGraph()
for (src, dst), weight in module_edges.items():
    if(weight < 30): continue
    G.add_edge(src, dst, weight=weight)

print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")

if len(G.edges()) > 0:
    plt.figure(figsize=(14, 12))
    
    # Use a different layout algorithm for better visualization
    pos = nx.kamada_kawai_layout(G)
    
    # Node sizes based on number of connections (in and out degree)
    node_sizes = [3000 + 100 * (G.out_degree(n) + G.in_degree(n)) for n in G.nodes()]
    
    # Edge widths based on weight
    max_edge_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
    edge_weights = [1 + 3 * G[u][v]['weight'] / max_edge_weight for u, v in G.edges()]
    
    # Node colors based on out degree (number of outgoing connections)
    node_colors = [G.out_degree(n) for n in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
            node_color=node_colors, cmap=plt.cm.viridis, 
            font_size=12, arrowsize=15, width=edge_weights, 
            edge_color='gray', font_weight='bold', arrows=True)
    
    # Only show edge labels for significant dependencies (weight > 5)
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges() if G[u][v]['weight'] > 5}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title('Zeeguu-Web Module Dependency Graph', fontsize=16)
    plt.axis('off')
    
    # Save the graph as an image file
    plt.savefig('zeeguu_dependency_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No edges in the graph. Cannot visualize an empty graph.")