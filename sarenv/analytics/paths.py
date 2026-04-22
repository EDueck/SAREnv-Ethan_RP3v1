# sarenv/analytics/paths.py
"""
Mathematical routing library containing the core spatial heuristics and path generation 
algorithms for multi-agent Unmanned Aerial Vehicle (UAV) search operations.
"""
import logging
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring


def split_path_for_drones(path: LineString, num_drones: int) -> list[LineString]:
    """
    Segments a continuous geometrical path into equidistant discrete trajectories, 
    allowing a singular global routing pattern to be executed by a multi-agent swarm.
    """
    if num_drones <= 1 or path.is_empty or path.length == 0:
        return [path]
    segments = []
    segment_length = path.length / num_drones
    for i in range(num_drones):
        segments.append(substring(path, i * segment_length, (i + 1) * segment_length))
    return segments


def generate_spiral_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float,
                         overlap: float, num_drones: int, path_point_spacing_m: float, **kwargs) -> list[LineString]:
    """
    Generates a continuous Archimedean spiral extending from the initial deployment coordinate.
    Calculates loop spacing dynamically based on the optical sensor footprint to ensure unbroken spatial coverage.
    """
    budget = kwargs.get('budget')
    
    # Derives the required translation between spiral loops based on the optical footprint
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    
    # Approximates the arc length to determine necessary discrete coordinate resolution
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max ** 2) + np.log(
        theta_max + np.sqrt(1 + theta_max ** 2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta), strict=True))
    paths = split_path_for_drones(full_path, num_drones)

    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)

    return paths


def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float,
                                     altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float,
                                     transition_distance_m: float, **kwargs) -> list[LineString]:
    """
    Generates a layered radial sweep pattern characterized by equidistant concentric circles,
    connected by discretized transition vectors to maintain continuous flight kinematics.
    """
    budget = kwargs.get('budget')
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        
        path_points.extend(
            zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta), strict=True))
        next_radius = current_radius + radius_increment
        
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(
                zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta),
                    strict=True))
        current_radius = next_radius
        
    full_path = LineString(path_points) if path_points else LineString()
    paths = split_path_for_drones(full_path, num_drones)

    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)

    return paths


def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float,
                               altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float,
                               **kwargs) -> list[LineString]:
    """
    Executes a sector-segmented sweeping heuristic. Divides the global search area into distinct 
    angular bounds to facilitate immediate parallel deployment of multiple independent agents.
    """
    budget = kwargs.get('budget')
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    
    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1
        
        while radius <= max_radius:
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle, end_angle = base_start_angle + angular_offset_rad, base_end_angle - angular_offset_rad
            
            if start_angle >= end_angle:
                radius += pass_width
                continue
                
            arc_length = radius * (end_angle - start_angle)
            num_arc_points = max(2, int(arc_length / path_point_spacing_m))
            current_arc_angles = np.linspace(start_angle, end_angle, num_arc_points) if direction == 1 else np.linspace(
                end_angle, start_angle, num_arc_points)
                
            points.extend(
                zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles),
                    strict=True))
            radius += pass_width
            direction *= -1
            
        if len(points) > 1:
            paths.append(LineString(points))

    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)

    return paths


def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map: np.ndarray, bounds: tuple,
                         max_radius: float, **kwargs) -> list[LineString]:
    """
    Executes a risk-aware greedy heuristic leveraging an 8-connected Moore neighbourhood matrix.
    Dynamically routes agents by evaluating the spatial probability gradient against the topological hazard penalty.
    """
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    rng = np.random.default_rng()

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    fov_deg = kwargs.get('fov_deg')
    altitude = kwargs.get('altitude')
    detection_radius = altitude * np.tan(np.radians(fov_deg / 2))

    risk_map = kwargs.get('risk_map')
    risk_bias = kwargs.get('risk_bias', 0.0)

    logging.getLogger("sarenv").debug(
        "Initializing greedy heuristic. Topographical risk map present: %s | Risk bias (\u03b2): %s", 
        risk_map is not None, risk_bias
    )

    # Computes the effective navigational gradient by blending the normalized probability and risk matrices
    if risk_map is not None and risk_bias > 0.0:
        prob_max = np.max(probability_map)
        norm_prob = probability_map / prob_max if prob_max > 0 else np.zeros_like(probability_map)

        risk_max = np.max(risk_map)
        norm_risk = risk_map / risk_max if risk_max > 0 else np.zeros_like(risk_map)

        effective_map = (1.0 - risk_bias) * norm_prob + risk_bias * norm_risk
    else:
        effective_map = probability_map

    # Pre-computes dimensional mappings to accelerate coordinate transformations during traversal
    dx = (maxx - minx) / width
    dy = ((maxy - miny) / height)
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    detection_radius_cells_x = int(np.ceil(detection_radius / dx))
    detection_radius_cells_y = int(np.ceil(detection_radius / dy))

    def get_visible_cells_from_grid_pos(row: int, col: int) -> set[tuple[int, int]]:
        """Calculates the discrete cell footprint captured by the optical sensor at a given position."""
        visible_cells = set()
        world_x = x_offset + col * dx
        world_y = y_offset + row * dy

        for r in range(max(0, row - detection_radius_cells_y),
                       min(height, row + detection_radius_cells_y + 1)):
            for c in range(max(0, col - detection_radius_cells_x),
                           min(width, col + detection_radius_cells_x + 1)):

                cell_x = x_offset + c * dx
                cell_y = y_offset + r * dy

                distance = np.sqrt((cell_x - world_x) ** 2 + (cell_y - world_y) ** 2)
                if distance <= detection_radius:
                    visible_cells.add((r, c))

        return visible_cells

    def calculate_position_score(row: int, col: int, observed_cells: set) -> float:
        """Evaluates the immediate heuristic yield of a spatial translation, omitting historically observed cells."""
        visible_cells = get_visible_cells_from_grid_pos(row, col)
        new_cells = visible_cells - observed_cells
        return sum(effective_map[r, c] for r, c in new_cells)

    start_col = np.clip(int((center_x - minx) / dx), 0, width - 1)
    start_row = np.clip(int((center_y - miny) / dy), 0, height - 1)
    start_pos = (start_row, start_col)

    max_radius_sq = max_radius * max_radius

    # Establishes the shared-memory stigmergy matrix to track swarm coverage globally
    globally_observed_cells = set()

    current_positions = [start_pos]
    for i in range(1, num_drones):
        angle = 2 * np.pi * i / num_drones
        offset_r = min(2, height // 10)
        offset_c = min(2, width // 10)
        new_r = np.clip(start_pos[0] + int(offset_r * np.sin(angle)), 0, height - 1)
        new_c = np.clip(start_pos[1] + int(offset_c * np.cos(angle)), 0, width - 1)
        current_positions.append((new_r, new_c))

    paths = [[] for _ in range(num_drones)]
    for i, pos in enumerate(current_positions):
        paths[i].append(pos)
        visible_cells = get_visible_cells_from_grid_pos(pos[0], pos[1])
        globally_observed_cells.update(visible_cells)

    # Defines the 8-connected Moore neighbourhood for cellular automaton spatial evaluation
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    max_iterations = height * width // num_drones

    budget = kwargs.get('budget')
    if budget is not None and budget > 0:
        max_iterations = (budget // dx) // num_drones

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        for i in range(num_drones):
            current_r, current_c = current_positions[i]
            valid_neighbors = []

            for dr, dc in neighbor_offsets:
                nr, nc = current_r + dr, current_c + dc
                
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                world_x = x_offset + nc * dx
                world_y = y_offset + nr * dy
                dist_sq = (world_x - center_x) ** 2 + (world_y - center_y) ** 2
                
                if dist_sq >= max_radius_sq:
                    continue

                score = calculate_position_score(nr, nc, globally_observed_cells)
                valid_neighbors.append(((nr, nc), score))

            if valid_neighbors:
                best_neighbor, max_prob = max(valid_neighbors, key=lambda x: x[1])

                # Reverts to uniform stochastic movement if the heuristic gradient collapses (local optima trap)
                if max_prob <= 0:
                    best_neighbor = rng.choice(len(valid_neighbors))
                    best_neighbor = valid_neighbors[best_neighbor][0]
                    
                current_positions[i] = best_neighbor
                paths[i].append(best_neighbor)

                visible_cells = get_visible_cells_from_grid_pos(best_neighbor[0], best_neighbor[1])
                globally_observed_cells.update(visible_cells)

    # Reconstructs the continuous geospatial LineStrings from the discrete cellular pathings
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            line_paths.append(LineString([(x_offset + c * dx, y_offset + r * dy) for r, c in drone_path_indices]))
        else:
            line_paths.append(LineString())

    if budget is not None and budget > 0:
        paths = restrict_path_length(line_paths, budget / num_drones)
    else:
        paths = line_paths

    return paths


def generate_random_walk_path(
        center_x: float,
        center_y: float,
        num_drones: int,
        probability_map,
        **kwargs
) -> list[LineString]:
    """
    Acts as the untrained statistical baseline by enforcing purely stochastic spatial navigation.
    Achieved by nullifying the heuristic matrices and feeding a zero-gradient array to the greedy architecture.
    """
    kwargs['risk_bias'] = 0.0
    return generate_greedy_path(
        center_x=center_x,
        center_y=center_y,
        num_drones=num_drones,
        probability_map=np.zeros_like(probability_map),
        **kwargs
    )


def restrict_path_length(line: LineString, max_length: float) -> LineString:
    """
    Truncates the generated spatial geometry to explicitly satisfy hardware operational flight limits.
    """
    if isinstance(line, list):
        return [restrict_path_length(path, max_length) for path in line]
    if line.is_empty or max_length is None or max_length <= 0 or line.length <= max_length:
        return line
    return substring(line, 0, max_length)
