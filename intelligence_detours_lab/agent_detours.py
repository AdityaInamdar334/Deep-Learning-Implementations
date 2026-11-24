"""Toy models of fixed behavior, replanning, and optionality. No claims about biology."""
from collections import deque
from dataclasses import dataclass

GRID = (7, 5)
START = (0, 2)
GOAL = (6, 2)
INITIAL_WALLS = {(3, 0), (3, 4)}
NEW_WALL = (3, 2)


def neighbors(point, walls=frozenset()):
    x, y = point
    width, height = GRID
    for nxt in ((x+1,y),(x,y-1),(x,y+1),(x-1,y)):
        if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in walls:
            yield nxt


def shortest_path(start, goal, walls):
    if start in walls or goal in walls:
        return None
    queue = deque([start]); previous = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current); current = previous[current]
            return path[::-1]
        for nxt in neighbors(current, walls):
            if nxt not in previous:
                previous[nxt] = current; queue.append(nxt)
    return None


def fixed_plan_agent(walls_before, walls_after):
    path = shortest_path(START, GOAL, walls_before)
    if path is None: return False, []
    walked = [path[0]]
    for nxt in path[1:]:
        if nxt in walls_after:
            return False, walked
        walked.append(nxt)
    return True, walked


def adaptive_agent(walls_before, walls_after):
    """Agent notices the new wall when its next step is blocked and replans."""
    path = shortest_path(START, GOAL, walls_before)
    if not path: return False, [], 0
    visited = [START]; cursor = START; replans = 0; known_walls = set(walls_before)
    while cursor != GOAL:
        if not path or len(path) < 2:
            return False, visited, replans
        nxt = path[1]
        if nxt in walls_after:
            known_walls.add(nxt)
            path = shortest_path(cursor, GOAL, known_walls)
            replans += 1
            if not path: return False, visited, replans
            continue
        cursor = nxt; visited.append(cursor); path = path[1:]
    return True, visited, replans


def reachable_state_count(start, walls, max_steps=3):
    """Count distinct states reachable in <= max_steps, including the start."""
    states = {start}; frontier = {start}
    for _ in range(max_steps):
        frontier = {nxt for state in frontier for nxt in neighbors(state, walls)}
        states |= frontier
    return len(states)


def draw(walls, path=()):
    active = set(path)
    rows = []
    for y in range(GRID[1]):
        rows.append(''.join('S' if (x,y)==START else 'G' if (x,y)==GOAL else '#' if (x,y) in walls else '*' if (x,y) in active else '.' for x in range(GRID[0])))
    return '\n'.join(rows)


def main():
    old = set(INITIAL_WALLS)
    new = old | {NEW_WALL}
    first = shortest_path(START, GOAL, old)
    fixed_ok, fixed_walk = fixed_plan_agent(old, new)
    adapt_ok, adapt_walk, replans = adaptive_agent(old, new)
    print('A simple world, after a new obstacle appears:')
    print(draw(new, adapt_walk))
    print('\nOriginal plan:', first)
    print('Fixed plan succeeds:', fixed_ok, '| stops at:', fixed_walk[-1])
    print('Adaptive plan succeeds:', adapt_ok, '| replans:', replans)
    print('Adaptive path:', adapt_walk)
    for pos in ((0,0),(2,2),(3,1)):
        print('Reachable within three moves from',pos,':',reachable_state_count(pos,new,3))
    print('\nThis models response flexibility and reachable options, NOT animal cognition, consciousness, or a general measure of intelligence.')

if __name__ == '__main__': main()
