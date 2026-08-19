import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import io, color
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
import networkx as nx
from collections import defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_maze(image_path):

    img = io.imread(image_path)

    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = color.rgb2gray(img[:, :, :3])
        else:
            gray = color.rgb2gray(img)

        if img.shape[2] >= 3:
            blue_mask = (img[:,:,2] > 100) & (img[:,:,0] < 150) & (img[:,:,1] < 150)
            white_mask = gray > 0.5
            free_mask = white_mask | blue_mask
        else:
            free_mask = gray > 0.5
    else:
        gray = img / 255.0 if img.max() > 1 else img
        free_mask = gray > 0.5

    free_mask = remove_small_holes(free_mask, area_threshold=500)
    free_mask = remove_small_objects(free_mask, min_size=500)

    return free_mask, img

def compute_distance_transform(free_mask):
    return ndimage.distance_transform_edt(free_mask)

def extract_skeleton_graph(free_mask):

    skeleton = skeletonize(free_mask)
    G = nx.Graph()

    skel_coords = np.argwhere(skeleton)
    for coord in skel_coords:
        G.add_node(tuple(coord))

    for coord in skel_coords:
        y, x = coord
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G.nodes:
                    weight = np.sqrt(dy**2 + dx**2)
                    G.add_edge((y, x), (ny, nx_), weight=weight)

    return skeleton, G

def find_bifurcation_points_clustered(G, cluster_radius=10, use_graph_distance=True):

    raw_bifurcations = [n for n in G.nodes if G.degree(n) >= 3]

    if not raw_bifurcations:
        return []

    bif_array = np.array(raw_bifurcations)
    n_bif = len(raw_bifurcations)

    if use_graph_distance:

        dist_matrix = np.full((n_bif, n_bif), np.inf)
        for i in range(n_bif):
            for j in range(i + 1, n_bif):
                try:
                    d = nx.dijkstra_path_length(G, raw_bifurcations[i], raw_bifurcations[j], weight='weight')
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                except nx.NetworkXNoPath:
                    pass
    else:

        dist_matrix = np.zeros((n_bif, n_bif))
        for i in range(n_bif):
            for j in range(i + 1, n_bif):
                d = np.sqrt(np.sum((bif_array[i] - bif_array[j])**2))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    parent = list(range(n_bif))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n_bif):
        for j in range(i + 1, n_bif):
            if dist_matrix[i, j] <= cluster_radius:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n_bif):
        clusters[find(i)].append(i)

    representative_bifurcations = []
    for cluster_indices in clusters.values():

        max_degree = -1
        best_idx = cluster_indices[0]

        for idx in cluster_indices:
            node = raw_bifurcations[idx]
            deg = G.degree(node)
            if deg > max_degree:
                max_degree = deg
                best_idx = idx

        representative_bifurcations.append(raw_bifurcations[best_idx])

    return representative_bifurcations

def find_endpoint_nodes_filtered(G, image_shape, boundary_margin=30):

    h, w = image_shape
    endpoints = []
    boundary_endpoints = []                               

    for node in G.nodes:
        if G.degree(node) == 1:
            y, x = node

            near_boundary = (y < boundary_margin or y > h - boundary_margin or
                           x < boundary_margin or x > w - boundary_margin)
            if near_boundary:
                boundary_endpoints.append(node)
            else:
                endpoints.append(node)

    return endpoints, boundary_endpoints

def get_path_neighbors(G, start_node, arc_length, direction_hint=None):

    current = start_node
    visited = {start_node}
    if direction_hint:
        visited.add(direction_hint)

    total_dist = 0

    while total_dist < arc_length:
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        if not neighbors:
            break

        next_node = neighbors[0]
        edge_dist = G[current][next_node]['weight']
        total_dist += edge_dist
        visited.add(next_node)
        current = next_node

    return current

def compute_curvature_at_point_v2(point, G, arc_length_window=30):

    if point not in G.nodes:
        return 0.0

    neighbors = list(G.neighbors(point))
    if len(neighbors) < 2:
        return 0.0

    p_minus = get_path_neighbors(G, point, arc_length_window, direction_hint=neighbors[-1])
    p_plus = get_path_neighbors(G, point, arc_length_window, direction_hint=neighbors[0])

    p0 = np.array(point, dtype=float)
    pm = np.array(p_minus, dtype=float)
    pp = np.array(p_plus, dtype=float)

    a = np.linalg.norm(pp - p0)
    b = np.linalg.norm(pm - p0)
    c = np.linalg.norm(pp - pm)

    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0

    area = 0.5 * abs((pp[0] - p0[0]) * (pm[1] - p0[1]) - (pm[0] - p0[0]) * (pp[1] - p0[1]))

    if area < 1e-6:
        return 0.0

    return 4 * area / (a * b * c)

def compute_curvature_field_v2(skeleton, G, arc_length_window=30, smooth_sigma=3):

    raw_curvatures = {}
    for node in G.nodes:
        raw_curvatures[node] = compute_curvature_at_point_v2(node, G, arc_length_window)

    if smooth_sigma <= 0:
        return raw_curvatures

    smoothed_curvatures = raw_curvatures.copy()

    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)

        endpoints = [n for n in subgraph.nodes if subgraph.degree(n) == 1]

        if len(endpoints) < 2:
            continue

        try:

            path = nx.dijkstra_path(subgraph, endpoints[0], endpoints[-1], weight='weight')

            if len(path) > 2 * smooth_sigma:
                curvs = np.array([raw_curvatures[p] for p in path])
                smoothed = gaussian_filter1d(curvs, sigma=smooth_sigma, mode='reflect')

                for i, p in enumerate(path):
                    smoothed_curvatures[p] = smoothed[i]
        except nx.NetworkXNoPath:
            continue

    return smoothed_curvatures

def project_to_skeleton(point, skeleton, search_radius=None):

    skel_coords = np.argwhere(skeleton)
    if len(skel_coords) == 0:
        return None

    distances = np.sqrt(np.sum((skel_coords - point)**2, axis=1))

    if search_radius is not None:
        valid = distances < search_radius
        if not np.any(valid):
            return None
        distances = np.where(valid, distances, np.inf)

    nearest_idx = np.argmin(distances)
    return tuple(skel_coords[nearest_idx])

def compute_geodesic_distance_field(skeleton, G, target_point):

    target_skel = project_to_skeleton(np.array(target_point), skeleton)

    if target_skel is None or target_skel not in G.nodes:
        return {}

    try:
        distances = nx.single_source_dijkstra_path_length(G, target_skel, weight='weight')
    except nx.NetworkXError:
        distances = {}
        for component in nx.connected_components(G):
            if target_skel in component:
                subgraph = G.subgraph(component)
                distances = nx.single_source_dijkstra_path_length(subgraph, target_skel, weight='weight')
                break

    return distances

def compute_bifurcation_distance_field(G, bifurcations):

    if not bifurcations:
        return {node: np.inf for node in G.nodes}

    bif_distances = {}
    for node in G.nodes:
        min_dist = np.inf
        for bif in bifurcations:
            try:
                dist = nx.dijkstra_path_length(G, node, bif, weight='weight')
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        bif_distances[node] = min_dist

    return bif_distances

class MazeFeatureExtractor:

    def __init__(self, free_mask, targets, start_point, pixel_scale=1.0,
                 bifurcation_cluster_radius=10,
                 boundary_margin=30,
                 curvature_arc_length=30,
                 curvature_smooth_sigma=3):

        self.free_mask = free_mask
        self.targets = targets
        self.start_point = start_point
        self.pixel_scale = pixel_scale
        self.shape = free_mask.shape

        self.params = {
            'bifurcation_cluster_radius': bifurcation_cluster_radius,
            'boundary_margin': boundary_margin,
            'curvature_arc_length': curvature_arc_length,
            'curvature_smooth_sigma': curvature_smooth_sigma
        }

        print("="*60)
        print("Maze Feature Extractor v3 (with fixes)")
        print("="*60)
        print(f"Parameters:")
        print(f"  - Bifurcation cluster radius: {bifurcation_cluster_radius} px")
        print(f"  - Boundary margin: {boundary_margin} px")
        print(f"  - Curvature arc length: {curvature_arc_length} px")
        print(f"  - Curvature smooth sigma: {curvature_smooth_sigma}")
        print("-"*60)

        print("\n[1/6] Computing distance transform...")
        self.dt = compute_distance_transform(free_mask)

        print("[2/6] Extracting skeleton...")
        self.skeleton, self.G = extract_skeleton_graph(free_mask)
        print(f"      Skeleton nodes: {len(self.G.nodes)}")

        print("[3/6] Finding special points (with clustering & filtering)...")

        raw_bif = [n for n in self.G.nodes if self.G.degree(n) >= 3]
        raw_end = [n for n in self.G.nodes if self.G.degree(n) == 1]

        self.bifurcations = find_bifurcation_points_clustered(
            self.G, cluster_radius=bifurcation_cluster_radius, use_graph_distance=True
        )
        self.endpoints, self.boundary_endpoints = find_endpoint_nodes_filtered(
            self.G, self.shape, boundary_margin=boundary_margin
        )

        print(f"      Raw bifurcations: {len(raw_bif)} → Clustered: {len(self.bifurcations)}")
        print(f"      Raw endpoints: {len(raw_end)} → Filtered: {len(self.endpoints)} "
              f"(removed {len(self.boundary_endpoints)} boundary)")

        print("[4/6] Computing geodesic distance fields...")
        self.geo_distances = {}
        for name, target in targets.items():
            self.geo_distances[name] = compute_geodesic_distance_field(
                self.skeleton, self.G, target
            )
            print(f"      Target {name}: {len(self.geo_distances[name])} reachable points")

        print("[5/6] Computing bifurcation distance field...")
        self.bif_distances = compute_bifurcation_distance_field(self.G, self.bifurcations)

        print("[6/6] Computing curvature field (arc-length window + smoothing)...")
        self.curvatures = compute_curvature_field_v2(
            self.skeleton, self.G,
            arc_length_window=curvature_arc_length,
            smooth_sigma=curvature_smooth_sigma
        )

        self._compute_normalization()
        print("\n✓ Precomputation complete!")
        print("="*60)

    def _compute_normalization(self):

        self.max_geo = {
            name: max(d.values()) if d else 1.0 
            for name, d in self.geo_distances.items()
        }
        self.max_dt = np.max(self.dt)
        finite_bif = [d for d in self.bif_distances.values() if np.isfinite(d)]
        self.max_bif = max(finite_bif) if finite_bif else 1.0
        self.max_curvature = max(self.curvatures.values()) if self.curvatures else 1.0

    def query(self, point, target_name, normalize=True):

        y, x = int(point[0]), int(point[1])
        y = max(0, min(y, self.shape[0] - 1))
        x = max(0, min(x, self.shape[1] - 1))

        skel_point = project_to_skeleton(np.array([y, x]), self.skeleton)

        D_geo = self.geo_distances.get(target_name, {}).get(skel_point, np.nan) if skel_point else np.nan
        d_wall = self.dt[y, x]
        kappa = self.curvatures.get(skel_point, 0.0) if skel_point else 0.0
        d_bif = self.bif_distances.get(skel_point, np.inf) if skel_point else np.inf
        width = 2 * self.dt[skel_point] if skel_point else 2 * d_wall

        if normalize:
            return {
                'D_geo': D_geo / self.max_geo.get(target_name, 1.0) if not np.isnan(D_geo) else np.nan,
                'd_wall': d_wall / self.max_dt,
                'kappa': kappa / self.max_curvature if self.max_curvature > 0 else 0,
                'd_bif': d_bif / self.max_bif if np.isfinite(d_bif) else 1.0,
                'width': width / (2 * self.max_dt)
            }
        else:
            scale = self.pixel_scale
            return {
                'D_geo': D_geo * scale,
                'd_wall': d_wall * scale,
                'kappa': kappa / scale,
                'd_bif': d_bif * scale if np.isfinite(d_bif) else np.inf,
                'width': width * scale
            }

    def query_vector(self, point, target_name, normalize=True):

        r = self.query(point, target_name, normalize)
        return np.array([r['D_geo'], r['d_wall'], r['kappa'], r['d_bif'], r['width']])

    def save(self, filepath):

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved to {filepath}")

    @staticmethod
    def load(filepath):

        with open(filepath, 'rb') as f:
            return pickle.load(f)

def create_comparison_visualization(extractor, original_img, save_dir):

    fig1, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(original_img)

    skel_coords = np.argwhere(extractor.skeleton)
    ax.scatter(skel_coords[:, 1], skel_coords[:, 0], c='cyan', s=0.3, alpha=0.6)

    if extractor.bifurcations:
        bif_arr = np.array(extractor.bifurcations)
        ax.scatter(bif_arr[:, 1], bif_arr[:, 0], c='red', s=120, marker='o', 
                  edgecolors='white', linewidths=2, label=f'Bifurcations (n={len(extractor.bifurcations)})', zorder=5)

    if extractor.endpoints:
        end_arr = np.array(extractor.endpoints)
        ax.scatter(end_arr[:, 1], end_arr[:, 0], c='lime', s=100, marker='^',
                  edgecolors='white', linewidths=2, label=f'Valid Endpoints (n={len(extractor.endpoints)})', zorder=5)

    if extractor.boundary_endpoints:
        bnd_arr = np.array(extractor.boundary_endpoints)
        ax.scatter(bnd_arr[:, 1], bnd_arr[:, 0], c='gray', s=60, marker='x',
                  alpha=0.5, label=f'Boundary Endpoints (excluded, n={len(extractor.boundary_endpoints)})', zorder=4)

    colors = {'A': 'yellow', 'B': 'orange', 'C': 'magenta'}
    for name, (y, x) in extractor.targets.items():
        ax.scatter(x, y, c=colors[name], s=250, marker='*', edgecolors='black', linewidths=2, zorder=6)
        ax.annotate(f'Target {name}', (x+15, y), fontsize=14, color=colors[name],
                   fontweight='bold', path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    sy, sx = extractor.start_point
    ax.scatter(sx, sy, c='white', s=350, marker='D', edgecolors='black', linewidths=2, zorder=6)
    ax.annotate('START', (sx+15, sy), fontsize=14, color='white', fontweight='bold',
               path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    h, w = extractor.shape
    margin = extractor.params['boundary_margin']
    rect = plt.Rectangle((margin, margin), w-2*margin, h-2*margin, 
                         fill=False, edgecolor='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(rect)

    ax.legend(loc='lower right', fontsize=11)
    ax.set_title(f'Skeleton Overview (v3)\n'
                f'Bifurcation cluster radius={extractor.params["bifurcation_cluster_radius"]}px, '
                f'Boundary margin={margin}px', fontsize=13)
    ax.axis('off')
    fig1.tight_layout()
    fig1.savefig(f'{save_dir}/01_skeleton_overview_v3.png', dpi=150, bbox_inches='tight')

    fig2, ax = plt.subplots(figsize=(10, 10))
    bif_img = np.full(extractor.shape, np.nan)
    for point, dist in extractor.bif_distances.items():
        if np.isfinite(dist):
            bif_img[point] = dist
    im = ax.imshow(bif_img, cmap='RdYlBu_r')
    plt.colorbar(im, ax=ax, label='Distance to bifurcation (pixels)', shrink=0.8)
    if extractor.bifurcations:
        bif_arr = np.array(extractor.bifurcations)
        ax.scatter(bif_arr[:, 1], bif_arr[:, 0], c='black', s=80, marker='o', 
                  edgecolors='white', linewidths=2, zorder=5)
    ax.set_title(f'Distance to Nearest Bifurcation (d_bif)\n'
                f'Clustered: {len(extractor.bifurcations)} bifurcation points', fontsize=13)
    ax.axis('off')
    fig2.tight_layout()
    fig2.savefig(f'{save_dir}/02_bifurcation_distance_v3.png', dpi=150, bbox_inches='tight')

    fig3, ax = plt.subplots(figsize=(10, 10))
    curv_img = np.full(extractor.shape, np.nan)
    for point, kappa in extractor.curvatures.items():
        curv_img[point] = kappa
    im = ax.imshow(curv_img, cmap='hot')
    plt.colorbar(im, ax=ax, label='Curvature κ (1/pixel)', shrink=0.8)
    ax.set_title(f'Curvature Field (κ) - Smoothed\n'
                f'Arc length={extractor.params["curvature_arc_length"]}px, '
                f'σ={extractor.params["curvature_smooth_sigma"]}', fontsize=13)
    ax.axis('off')
    fig3.tight_layout()
    fig3.savefig(f'{save_dir}/03_curvature_v3.png', dpi=150, bbox_inches='tight')

    fig4 = create_path_analysis_v3(extractor, 'A', save_dir)

    plt.close('all')
    return fig1, fig2, fig3, fig4

def create_path_analysis_v3(extractor, target_name, save_dir):

    start_skel = project_to_skeleton(np.array(extractor.start_point), extractor.skeleton)
    target_skel = project_to_skeleton(np.array(extractor.targets[target_name]), extractor.skeleton)

    try:
        path = nx.dijkstra_path(extractor.G, start_skel, target_skel, weight='weight')
    except:
        print("Could not find path")
        return None

    n_samples = min(100, len(path))                                    
    indices = np.linspace(0, len(path)-1, n_samples).astype(int)
    sample_points = [path[i] for i in indices]

    features = {'D_geo': [], 'd_wall': [], 'kappa': [], 'd_bif': [], 'width': []}
    distances_along_path = []
    cumulative_dist = 0

    for i, pt in enumerate(sample_points):
        f = extractor.query(pt, target_name, normalize=True)
        for key in features:
            features[key].append(f[key])

        if i > 0:
            prev_pt = sample_points[i-1]
            cumulative_dist += np.sqrt((pt[0]-prev_pt[0])**2 + (pt[1]-prev_pt[1])**2)
        distances_along_path.append(cumulative_dist)

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

    configs = [
        ('D_geo', 'Geodesic Distance to Target', 'blue', '到目标距离（单调递减 ✓）'),
        ('d_wall', 'Distance to Wall', 'green', '离壁距离'),
        ('kappa', 'Curvature (smoothed)', 'red', '曲率（平滑后，弯道处抬高）'),
        ('d_bif', 'Distance to Bifurcation', 'purple', '到分叉距离（接近分叉时降低）'),
        ('width', 'Channel Width', 'orange', '通道宽度')
    ]

    for ax, (key, title_en, color, title_cn) in zip(axes, configs):
        ax.plot(distances_along_path, features[key], color=color, linewidth=2.5)
        ax.fill_between(distances_along_path, features[key], alpha=0.25, color=color)
        ax.set_ylabel(key, fontsize=12)
        ax.set_title(f'{title_en} - {title_cn}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    axes[-1].set_xlabel('Distance along path (pixels)', fontsize=12)
    fig.suptitle(f'Feature Profile: START → Target {target_name} (v3 with fixes)', 
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{save_dir}/04_path_analysis_{target_name}_v3.png', dpi=150, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    import os
    from ..paths import ASSET_ROOT, OUTPUT_ROOT

    output_dir = str(OUTPUT_ROOT / "maze_features")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading maze...")
    free_mask, original_img = load_and_preprocess_maze(str(ASSET_ROOT / "maze.png"))
    print(f"Image size: {free_mask.shape}")

    start_point = (75, 75)
    targets = {
        'A': (800, 580),
        'B': (480, 800),
        'C': (130, 720),
    }

    extractor = MazeFeatureExtractor(
        free_mask, targets, start_point, 
        pixel_scale=1.0,
        bifurcation_cluster_radius=25,                                                   
        boundary_margin=30,                                                     
        curvature_arc_length=30,                                        
        curvature_smooth_sigma=3                                   
    )

    extractor.save(f'{output_dir}/maze_extractor_v3.pkl')

    print("\nGenerating visualizations...")
    create_comparison_visualization(extractor, original_img, output_dir)

    print("\n" + "="*60)
    print("DEMO: Feature queries")
    print("="*60)

    test_points = [
        ('Start', start_point),
        ('Mid-path', (400, 300)),
        ('Near bifurcation', (430, 480)),
    ]

    for name, pt in test_points:
        print(f"\n📍 {name} @ {pt}")
        for target in ['A', 'B', 'C']:
            v = extractor.query_vector(pt, target, normalize=True)
            print(f"   → Target {target}: D_geo={v[0]:.3f}, d_wall={v[1]:.3f}, "
                  f"κ={v[2]:.3f}, d_bif={v[3]:.3f}, w={v[4]:.3f}")

    print("\n" + "="*60)
    print(f"✓ All outputs saved to {output_dir}/")
    print("="*60)
