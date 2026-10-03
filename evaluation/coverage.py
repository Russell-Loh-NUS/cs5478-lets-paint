"""Coverage evaluation metric."""


def coverage_percentage(wall_map):
    total = wall_map.total_cells()
    if total == 0:
        return 0.0
    return 100.0 * wall_map.painted_count() / total


def coverage_error(wall_map):
    """Percentage of paintable cells left unpainted, plus a penalty for any obstacle
    (window/door) cells that were mistakenly painted."""
    total = wall_map.total_cells()
    if total == 0:
        return 0.0
    unpainted = total - wall_map.painted_count()
    violations = wall_map.painted_obstacle_count()
    return 100.0 * (unpainted + violations) / total
