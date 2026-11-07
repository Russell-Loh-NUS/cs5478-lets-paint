"""Pure wall-coordinate geometry; no neural actions or Bullet dependencies."""
import heapq
import math
import numpy as np


def intersects(a, b, rect):
    """Closed segment intersects the interior of axis-aligned rectangle."""
    lo, hi = 0., 1.
    for x, y, lower, upper in zip(a, b, rect[::2], rect[1::2]):
        lower += 1e-9
        upper -= 1e-9
        d = y-x
        if abs(d) < 1e-12:
            if not lower < x < upper:
                return False
        else:
            t0, t1 = sorted(((lower-x)/d, (upper-x)/d))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return False
    return True


def detour(start, end, rect, bounds):
    """Shortest visibility path around a single inflated rectangle."""
    nodes = [tuple(start), tuple(end)] + [(u,v) for u in rect[:2] for v in rect[2:]]
    valid = lambda q: bounds[0] <= q[0] <= bounds[1] and bounds[2] <= q[1] <= bounds[3]
    if not valid(nodes[0]) or not valid(nodes[1]):
        raise RuntimeError('route_unreachable: endpoint outside wall clearance bounds')
    queue, seen = [(0., 0, [0])], set()
    while queue:
        distance, i, path = heapq.heappop(queue)
        if i in seen:
            continue
        seen.add(i)
        if i == 1:
            return [np.array(nodes[j], float) for j in path[1:]]
        for j, point in enumerate(nodes):
            if j not in seen and valid(point) and not intersects(nodes[i], point, rect):
                heapq.heappush(queue, (distance+math.dist(nodes[i],point),j,path+[j]))
    raise RuntimeError('route_unreachable: no wall-plane detour')


def sweep_segments(bounds, rect, feasible, pitch=.075, spacing=.02):
    """Split vertical lanes at the exclusion and exact IK failures."""
    u0,u1,v0,v1 = bounds
    segments = []
    for index,u in enumerate(np.linspace(u0,u1,max(2,math.ceil((u1-u0)/pitch)+1))):
        vs = np.linspace(v0,v1,max(2,math.ceil((v1-v0)/spacing)+1))
        runs, run = [], []
        for v in vs:
            point = np.array([u,v])
            if not intersects(point,point,rect) and feasible(point):
                run.append(point)
            else:
                if len(run)>1: runs.append((run[0],run[-1]))
                run=[]
        if len(run)>1: runs.append((run[0],run[-1]))
        if index % 2: runs = [(b,a) for a,b in reversed(runs)]
        segments.extend(runs)
    if not segments:
        raise RuntimeError('route_unreachable: no valid painting segments')
    return segments
