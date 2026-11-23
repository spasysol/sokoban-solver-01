// Sokoban solver for make.fun's sokoban-solver-01-run-20261121 bounty.
//
// Strategy: the generator lays a hallway across the board with 1-4 rooms off
// it, each room holding 3-4 boxes and as many goals, opening onto the hallway
// through one door, and a box never leaves its room. So the level's push
// optimum P* is the sum of the rooms' optima, and this program solves each
// room independently and push-optimally, then stitches the rooms together
// with walks through the hallway.
//
// Each room is solved by A* over pushes (not moves): a state is the boxes'
// positions plus the player's reachable region, each push costs 1, and the
// estimate is the cheapest assignment of boxes to goals by Manhattan
// distance. Dead squares (cells no box can be pushed from to a goal) and
// 2x2 blocks of boxes prune the search. The player walks between pushes by
// flood fill, so the move string is the walks plus the pushes.
//
// Standard library only; no network, no clock, no randomness. The search
// budget is a fixed number of node expansions per room, so two runs always
// write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// One wall budget of work per room, counted in node expansions, never time.
const ROOM_CAP = 200000;

// Directions: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

type Grid = {
  w: number;
  h: number;
  wall: boolean[];
  goals: Set<number>;
  boxes: number[];
  player: number;
};

function parse(text: string): Grid {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(false);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') wall[i] = true;
      if ('.*+'.includes(ch)) goals.add(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

const idx = (g: Grid, x: number, y: number): number => y * g.w + x;
const inBounds = (g: Grid, x: number, y: number): boolean => x >= 0 && x < g.w && y >= 0 && y < g.h;

// Flood fill from start over cells where open(i) is true; returns the set.
function reach(g: Grid, start: number, open: (i: number) => boolean): Set<number> {
  const seen = new Set<number>([start]);
  const stack = [start];
  while (stack.length > 0) {
    const c = stack.pop()!;
    const x = c % g.w;
    const y = Math.floor(c / g.w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (!inBounds(g, nx, ny)) continue;
      const n = idx(g, nx, ny);
      if (seen.has(n) || !open(n)) continue;
      seen.add(n);
      stack.push(n);
    }
  }
  return seen;
}

// Shortest walk from start to target over open cells, as a move string.
// Empty string when start equals target; null when unreachable.
function walk(g: Grid, start: number, target: number, open: (i: number) => boolean): string | null {
  if (start === target) return '';
  const prev = new Map<number, [number, string]>();
  const queue = [start];
  prev.set(start, [-1, '']);
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++]!;
    const x = c % g.w;
    const y = Math.floor(c / g.w);
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d]!;
      const ny = y + DY[d]!;
      if (!inBounds(g, nx, ny)) continue;
      const n = idx(g, nx, ny);
      if (prev.has(n) || !open(n)) continue;
      prev.set(n, [c, CH[d]!]);
      if (n === target) {
        let s = '';
        let cur = n;
        while (cur !== start) {
          const [p, ch] = prev.get(cur)!;
          s = ch + s;
          cur = p;
        }
        return s;
      }
      queue.push(n);
    }
  }
  return null;
}

type Room = {
  cells: Set<number>; // room component cells, door included
  boxes: number[];
  goals: number[];
  door: number; // component cell adjacent to the hallway
  approach: number; // hallway cell next to the door
};

// Split the level into rooms: the hallway is the player's row; each room is
// a connected component of free cells off that row holding boxes.
function roomsOf(g: Grid): Room[] | null {
  const H = Math.floor(g.player / g.w);
  const seen = new Set<number>();
  const rooms: Room[] = [];
  const free = (i: number): boolean => !g.wall[i];
  for (let y = 0; y < g.h; y++) {
    for (let x = 0; x < g.w; x++) {
      const i = idx(g, x, y);
      if (y === H || !free(i) || seen.has(i)) continue;
      // A new component: flood fill within rows other than H.
      const comp = reach(g, i, (c) => free(c) && Math.floor(c / g.w) !== H);
      for (const c of comp) seen.add(c);
      const boxes = g.boxes.filter((b) => comp.has(b)).sort((a, b) => a - b);
      if (boxes.length === 0) continue;
      const goals = [...g.goals].filter((c) => comp.has(c)).sort((a, b) => a - b);
      if (boxes.length !== goals.length || boxes.length < 3 || boxes.length > 4) return null;
      // The door is the component cell next to a hallway cell.
      let door = -1;
      let approach = -1;
      for (const c of comp) {
        const cx = c % g.w;
        const cy = Math.floor(c / g.w);
        for (let d = 0; d < 4; d++) {
          const nx = cx + DX[d]!;
          const ny = cy + DY[d]!;
          if (!inBounds(g, nx, ny) || ny !== H) continue;
          const n = idx(g, nx, ny);
          if (!free(n)) continue;
          if (door === -1 || c < door) {
            door = c;
            approach = n;
          }
        }
      }
      if (door === -1) return null;
      rooms.push({ cells: comp, boxes, goals, door, approach });
    }
  }
  const covered = rooms.reduce((a, r) => a + r.boxes.length, 0);
  if (covered !== g.boxes.length || rooms.length === 0) return null;
  rooms.sort((a, b) => a.approach - b.approach);
  return rooms;
}

// Cells a box can be pushed from to reach a goal: reverse search from the
// goals. A cell x is live when some push from x lands on a live cell and the
// player can stand on the far side to make it.
function liveCells(g: Grid, cells: Set<number>, goals: number[]): Set<number> {
  const live = new Set<number>(goals);
  let changed = true;
  while (changed) {
    changed = false;
    for (const x of cells) {
      if (live.has(x)) continue;
      const xx = x % g.w;
      const xy = Math.floor(x / g.w);
      for (let d = 0; d < 4; d++) {
        const ax = xx + DX[d]!;
        const ay = xy + DY[d]!;
        const bx = xx - DX[d]!;
        const by = xy - DY[d]!;
        if (!inBounds(g, ax, ay) || !inBounds(g, bx, by)) continue;
        const ahead = idx(g, ax, ay);
        const behind = idx(g, bx, by);
        if (live.has(ahead) && cells.has(behind)) {
          live.add(x);
          changed = true;
          break;
        }
      }
    }
  }
  return live;
}

// Cheapest assignment of boxes to goals by Manhattan distance (n <= 4, so
// trying every permutation is cheap). Admissible: each box needs at least
// its Manhattan distance in pushes to reach its goal.
function estimate(g: Grid, boxes: number[], goals: number[]): number {
  const n = boxes.length;
  const order = goals.map((_, i) => i);
  let best = Infinity;
  const perm = (arr: number[], k: number): void => {
    if (k === n) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const b = boxes[i]!;
        const gl = goals[arr[i]!]!;
        s += Math.abs((b % g.w) - (gl % g.w)) + Math.abs(Math.floor(b / g.w) - Math.floor(gl / g.w));
        if (s >= best) return;
      }
      if (s < best) best = s;
      return;
    }
    for (let i = k; i < n; i++) {
      [arr[k], arr[i]] = [arr[i]!, arr[k]!];
      perm(arr, k + 1);
      [arr[k], arr[i]] = [arr[i]!, arr[k]!];
    }
  };
  perm(order, 0);
  return best;
}

// A 2x2 block of boxes with any box off a goal is frozen: no box in it can move.
function frozen2x2(g: Grid, boxes: Set<number>, goals: Set<number>): boolean {
  for (const b of boxes) {
    const x = b % g.w;
    const y = Math.floor(b / g.w);
    if (x + 1 >= g.w || y + 1 >= g.h) continue;
    const c2 = idx(g, x + 1, y);
    const c3 = idx(g, x, y + 1);
    const c4 = idx(g, x + 1, y + 1);
    if (boxes.has(c2) && boxes.has(c3) && boxes.has(c4)) {
      if (!goals.has(b) || !goals.has(c2) || !goals.has(c3) || !goals.has(c4)) return true;
    }
  }
  return false;
}

// Push-optimal A* for one room, with a fixed node budget. The player starts
// on the approach cell (the hallway side of the door). Returns the moves
// from the approach cell back to the approach cell, or null when the room is
// not solved; exhausted is true when the budget, not the search, gave up.
function solveRoomOnce(g: Grid, room: Room, cap: number): { moves: string | null; exhausted: boolean } {
  const walkable = new Set<number>(room.cells);
  walkable.add(room.approach);
  const goalSet = new Set<number>(room.goals);
  const live = liveCells(g, room.cells, room.goals);
  const inWalk = (i: number): boolean => walkable.has(i);

  // State: sorted boxes + canonical player (lowest cell of the player's
  // reachable region). The open list is a heap on pushes + estimate.
  const boxes0 = [...room.boxes];
  const startReach = reach(g, room.approach, (c) => inWalk(c) && !boxes0.includes(c));
  const startPlayer = Math.min(...startReach);
  const keyOf = (bs: number[], p: number): string => `${bs.join(',')}|${p}`;
  const sBoxes: number[][] = [boxes0];
  const sPlayer: number[] = [startPlayer];
  const sG: number[] = [0];
  const parent: number[] = [-1];
  const pushBox: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<string, number>([[keyOf(boxes0, startPlayer), 0]]);
  const heap: [number, number][] = [[estimate(g, boxes0, room.goals), 0]];
  const less = (a: [number, number], b: [number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const heapPush = (e: [number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let goalState = -1;
  while (heap.length > 0 && expanded < cap) {
    const [, s] = heapPop();
    const bs = sBoxes[s]!;
    const p = sPlayer[s]!;
    if (best.get(keyOf(bs, p)) !== s) continue;
    expanded++;
    // Goal: every box on a goal and the player able to reach the door to leave.
    if (bs.every((b) => goalSet.has(b))) {
      const region = reach(g, p, (c) => inWalk(c) && !bs.includes(c));
      if (region.has(room.door)) {
        goalState = s;
        break;
      }
    }
    const boxSet = new Set<number>(bs);
    const region = reach(g, p, (c) => inWalk(c) && !boxSet.has(c));
    for (let k = 0; k < bs.length; k++) {
      const b = bs[k]!;
      const bx = b % g.w;
      const by = Math.floor(b / g.w);
      for (let d = 0; d < 4; d++) {
        const tx = bx + DX[d]!;
        const ty = by + DY[d]!;
        const px = bx - DX[d]!;
        const py = by - DY[d]!;
        if (!inBounds(g, tx, ty) || !inBounds(g, px, py)) continue;
        const target = idx(g, tx, ty);
        const pusher = idx(g, px, py);
        if (!room.cells.has(target) || boxSet.has(target)) continue;
        if (!region.has(pusher)) continue;
        const next = [...bs];
        next[k] = target;
        next.sort((a, bb) => a - bb);
        const nextSet = new Set<number>(next);
        if (next.some((c) => !live.has(c))) continue;
        if (frozen2x2(g, nextSet, goalSet)) continue;
        // The player steps into the box's old cell; normalise to the
        // lowest cell of the region it can reach from there.
        const after = reach(g, b, (c) => inWalk(c) && !nextSet.has(c));
        const np = Math.min(...after);
        const key = keyOf(next, np);
        const ng = sG[s]! + 1;
        if (best.has(key) && best.get(key)! <= ng) continue;
        const id = sBoxes.length;
        best.set(key, id);
        sBoxes.push(next);
        sPlayer.push(np);
        sG.push(ng);
        parent.push(s);
        pushBox.push(b);
        pushDir.push(d);
        heapPush([ng + estimate(g, next, room.goals), id]);
      }
    }
  }
  if (goalState === -1) return { moves: null, exhausted: expanded >= cap };

  // Rebuild the push list, then replay it forward, filling in the walks.
  const pushes: [number, number][] = [];
  let cur = goalState;
  while (cur > 0) {
    pushes.push([pushBox[cur]!, pushDir[cur]!]);
    cur = parent[cur]!;
  }
  pushes.reverse();
  let moves = '';
  let boxes = [...room.boxes];
  let player = room.approach;
  for (const [box, d] of pushes) {
    const bx = box % g.w;
    const by = Math.floor(box / g.w);
    const pusher = idx(g, bx - DX[d]!, by - DY[d]!);
    const boxSet = new Set<number>(boxes);
    const w = walk(g, player, pusher, (c) => inWalk(c) && !boxSet.has(c));
    if (w === null) return { moves: null, exhausted: false };
    moves += w + CH[d]!;
    const k = boxes.indexOf(box);
    boxes[k] = idx(g, bx + DX[d]!, by + DY[d]!);
    boxes.sort((a, b) => a - b);
    player = box;
  }
  // Walk out through the door and back onto the hallway.
  const boxSet = new Set<number>(boxes);
  const out = walk(g, player, room.door, (c) => inWalk(c) && !boxSet.has(c));
  if (out === null) return { moves: null, exhausted: false };
  moves += out;
  const dx = room.approach % g.w;
  const dy = Math.floor(room.approach / g.w);
  const ox = room.door % g.w;
  const oy = Math.floor(room.door / g.w);
  for (let d = 0; d < 4; d++) {
    if (ox + DX[d]! === dx && oy + DY[d]! === dy) moves += CH[d]!;
  }
  return { moves, exhausted: false };
}

// Solve one room, retrying once with a doubled budget when the first search
// runs out of nodes: a harder room in the scored set gets a second chance
// instead of an empty answer. The budget is still counted in work, never time.
function solveRoom(g: Grid, room: Room): string | null {
  const first = solveRoomOnce(g, room, ROOM_CAP);
  if (first.moves !== null || !first.exhausted) return first.moves;
  return solveRoomOnce(g, room, ROOM_CAP * 2).moves;
}

// Solve one level: walk the hallway from room to room, solving each in turn.
function solveLevel(text: string): string {
  const g = parse(text);
  const rooms = roomsOf(g);
  if (rooms === null) return '';
  // Track the boxes' current cells for the hallway walks.
  let boxes = [...g.boxes];
  let player = g.player;
  let moves = '';
  for (const room of rooms) {
    const boxSet = new Set<number>(boxes);
    const toDoor = walk(g, player, room.approach, (c) => !g.wall[c] && !boxSet.has(c));
    if (toDoor === null) return '';
    moves += toDoor;
    player = room.approach;
    const roomMoves = solveRoom(g, room);
    if (roomMoves === null) return '';
    moves += roomMoves;
    // This room's boxes are now on its goals.
    boxes = boxes.filter((b) => !room.cells.has(b));
    boxes.push(...room.goals);
    boxes.sort((a, b) => a - b);
    player = room.approach;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
