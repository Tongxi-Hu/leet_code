use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
    i32,
};

use crate::common::ListNode;

/// 01
pub fn average_waiting_time(customers: Vec<Vec<i32>>) -> f64 {
    let (wait, _) = customers
        .iter()
        .fold((0.0, customers[0][0] as f64), |mut acc, cur| {
            let (arr, cost) = (cur[0] as f64, cur[1] as f64);
            let time = if acc.1 < arr { arr } else { acc.1 } + cost;
            let wait = time - arr;
            acc.0 += wait;
            (acc.0, time)
        });
    wait / customers.len() as f64
}

/// 02
pub fn maximum_binary_string(binary: String) -> String {
    let n = binary.len();
    let mut s: Vec<char> = binary.chars().collect();
    let mut j = 0;
    for i in 0..n {
        if s[i] == '0' {
            while j <= i || (j < n && s[j] == '1') {
                j += 1;
            }
            if j < n {
                s[j] = '1';
                s[i] = '1';
                s[i + 1] = '0';
            }
        }
    }
    s.iter().collect()
}

/// 03
pub fn min_moves(mut nums: Vec<i32>, k: i32) -> i32 {
    let mut start = 0;
    for curr in 0..nums.len() {
        if nums[curr] == 1 {
            nums[start] = curr as i32;
            start += 1
        }
    }
    nums.truncate(start as usize);
    let mut iter = nums.windows(k as usize);
    let tmp = iter.next().unwrap();
    let currsum: i32 = tmp.iter().map(|x| (x - tmp[(k / 2) as usize]).abs()).sum();
    let c = iter.fold(currsum, |s, tmp| {
        s.min(tmp.iter().map(|x| (x - tmp[(k / 2) as usize]).abs()).sum())
    });
    c - (k as u64 * k as u64 / 4) as i32
}

/// 04
pub fn halves_are_alike(s: String) -> bool {
    let chars = s.chars().collect::<Vec<char>>();
    let vows = HashSet::from(['A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u']);
    let (mut l, mut r, mut cnt) = (0, chars.len() - 1, 0);
    while l < r {
        if vows.contains(&chars[l]) {
            cnt += 1;
        }
        if vows.contains(&chars[r]) {
            cnt -= 1;
        }
        l += 1;
        r -= 1;
    }
    cnt == 0
}

/// 05
pub fn eaten_apples(apples: Vec<i32>, days: Vec<i32>) -> i32 {
    let mut ans = 0;
    let mut pq = BinaryHeap::new();
    let n = apples.len();
    let mut i = 0;

    while i < n {
        while let Some(Reverse((rotten_day, _))) = pq.peek() {
            if *rotten_day <= i as i32 {
                pq.pop();
            } else {
                break;
            }
        }
        let rotten_day = i as i32 + days[i];
        let count = apples[i];
        if count > 0 {
            pq.push(Reverse((rotten_day, count)));
        }
        if let Some(Reverse((rotten_day, mut count))) = pq.pop() {
            count -= 1;
            if count > 0 {
                pq.push(Reverse((rotten_day, count)));
            }
            ans += 1;
        }
        i += 1;
    }

    while let Some(Reverse((rotten_day, count))) = pq.pop() {
        if rotten_day <= i as i32 {
            continue;
        }
        let num = std::cmp::min(rotten_day - i as i32, count);
        ans += num;
        i += num as usize;
    }
    ans
}

/// 06
pub fn find_ball(grid: Vec<Vec<i32>>) -> Vec<i32> {
    let n = grid[0].len();
    let mut ans = vec![0; n];
    for j in 0..n {
        let mut cur_col = j as i32;
        for row in &grid {
            let d = row[cur_col as usize];
            cur_col += d;
            if cur_col < 0 || cur_col as usize == n || row[cur_col as usize] != d {
                cur_col = -1;
                break;
            }
        }
        ans[j] = cur_col;
    }
    ans
}

/// 07
pub fn maximize_xor(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    struct TrieNode {
        next: Vec<Option<TrieNode>>,
    }

    let mut nums = nums;
    nums.sort_unstable();
    let mut queries = queries;
    queries
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| v.push(i as i32));
    queries.sort_unstable_by(|a, b| a[1].cmp(&b[1]));

    let mut trie = TrieNode {
        next: vec![None, None],
    };

    let mut nums_i = 0;
    for i in 0..queries.len() {
        if queries[i][1] < nums[0] {
            queries[i].push(-1);
            continue;
        }
        while nums_i < nums.len() && queries[i][1] >= nums[nums_i] {
            let mut trie_ptr = &mut trie;
            for nums_j in (0..32).rev() {
                let bit = ((nums[nums_i] >> nums_j) & 1) as usize;
                if trie_ptr.next[bit].is_none() {
                    trie_ptr.next[bit] = Some(TrieNode {
                        next: vec![None, None],
                    });
                }
                trie_ptr = trie_ptr.next[bit].as_mut().unwrap();
            }
            nums_i += 1;
        }

        let x = queries[i][0];
        let mut trie_ptr = &trie;
        let mut y = 0;
        for j in (0..32).rev() {
            let best = 1 - ((x >> j) & 1) as usize;
            if trie_ptr.next[best].is_none() {
                y = y | ((1 - best) << j) as i32;
                trie_ptr = trie_ptr.next[1 - best].as_ref().unwrap();
            } else {
                y = y | (best << j) as i32;
                trie_ptr = trie_ptr.next[best].as_ref().unwrap();
            }
        }
        queries[i].push(x ^ y);
    }
    queries.sort_unstable_by(|a, b| a[2].cmp(&b[2]));
    queries.into_iter().map(|x| x[3]).collect()
}

/// 10
pub fn maximum_units(mut box_types: Vec<Vec<i32>>, mut truck_size: i32) -> i32 {
    box_types.sort_by(|a, b| b[1].cmp(&a[1]));
    let mut total = 0;
    for i in 0..box_types.len() {
        if truck_size >= box_types[i][0] {
            total += box_types[i][0] * box_types[i][1];
            truck_size -= box_types[i][0];
        } else {
            total += truck_size * box_types[i][1];
            break;
        }
    }
    total
}

/// 11
pub fn count_pairs(deliciousness: Vec<i32>) -> i32 {
    let m = 10_i32.pow(9) + 7;
    let mut hashmap = HashMap::new();
    let pows: Vec<i32> = (0..22).map(|i| 2_i32.pow(i)).collect();
    let mut answer = 0;
    for x in deliciousness {
        for i in 0..22 {
            if pows[i] >= x {
                let y = pows[i] - x;
                if let Some(&n) = hashmap.get(&y) {
                    answer += n;
                    answer = answer % m;
                }
            }
        }
        *hashmap.entry(x).or_insert(0) += 1;
    }
    answer
}

/// 12
pub fn ways_to_split(mut nums: Vec<i32>) -> i32 {
    let mut ret = 0;

    for i in 1..nums.len() {
        nums[i] += nums[i - 1];
    }

    let sum = *nums.last().unwrap();

    for i in 0..nums.len() - 2 {
        let j = match nums[i + 1..].binary_search(&(2 * nums[i] - 1)) {
            Ok(a) => a + 1,
            Err(b) => b,
        };
        let k = match nums[i + 1..].binary_search(&((sum - nums[i]) / 2 + nums[i])) {
            Ok(a) if a == nums.len() - i - 2 => a,
            Ok(a) => a + 1,
            Err(b) if b == nums.len() - i - 1 => b - 1,
            Err(b) => b,
        };

        ret = (ret + k.saturating_sub(j) as i32) % 1_000_000_007;
    }

    ret
}

/// 13
pub fn min_operations(target: Vec<i32>, arr: Vec<i32>) -> i32 {
    let mp = target
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut acc, (i, v)| {
            acc.insert(v, i);
            acc
        });

    let mut vec = vec![];

    for val in arr.iter() {
        if let Some(&i) = mp.get(val) {
            let j = vec.binary_search(&i).unwrap_or_else(|x| x);
            if j == vec.len() {
                vec.push(i);
            } else {
                vec[j] = i;
            }
        }
    }

    (target.len() - vec.len()) as i32
}

/// 16
pub fn total_money(n: i32) -> i32 {
    const D: i32 = 7;
    let w = n / D;
    let r = n % D;
    (w * D * (w + D) + r * (w * 2 + r + 1)) / 2
}

/// 17
pub fn maximum_gain(s: String, x: i32, y: i32) -> i32 {
    let (a, b, a_s, b_s) = if x > y {
        (b'a', b'b', x, y)
    } else {
        (b'b', b'a', y, x)
    };
    let mut res = 0;
    let mut s = s.as_bytes().to_vec();
    fn remove_s(s: &mut Vec<u8>, a: u8, b: u8, sc: i32) -> i32 {
        let mut i = 0;
        let mut score = 0;
        while i + 1 < s.len() {
            if s[i] == a && s[i + 1] == b {
                score += sc;
                s.remove(i);
                s.remove(i);
                i = if i == 0 { 0 } else { i - 1 };
            } else {
                i += 1;
            }
        }
        score
    }
    res += remove_s(&mut s, a, b, a_s);
    res += remove_s(&mut s, b, a, b_s);
    res
}

/// 18
pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
    fn backtrace(n: i32, idx: usize, path: &mut Vec<i32>, vis: &mut HashSet<i32>) -> bool {
        if path.len() <= idx {
            true
        } else if path[idx] != 0 {
            backtrace(n, idx + 1, path, vis)
        } else {
            for x in (1..=n).rev() {
                if vis.insert(x) {
                    if x == 1 {
                        path[idx] = x;
                        if backtrace(n, idx + 1, path, vis) {
                            return true;
                        }
                        path[idx] = 0;
                    } else if matches!(path.get(idx + (x as usize)), Some(v) if *v == 0) {
                        path[idx] = x;
                        path[idx + (x as usize)] = x;
                        if backtrace(n, idx + 1, path, vis) {
                            return true;
                        }
                        path[idx] = 0;
                        path[idx + (x as usize)] = 0;
                    }
                    vis.remove(&x);
                }
            }
            false
        }
    }

    let mut vis = HashSet::new();
    let mut path = vec![0; n as usize * 2 - 1];
    backtrace(n, 0, &mut path, &mut vis);
    path
}

/// 19
pub fn check_ways(pairs: Vec<Vec<i32>>) -> i32 {
    let mut graph: HashMap<i32, HashSet<i32>> = HashMap::new();
    let mut result = 1;
    for pair in pairs {
        (*graph.entry(pair[0]).or_insert(HashSet::new())).insert(pair[1]);
        (*graph.entry(pair[1]).or_insert(HashSet::new())).insert(pair[0]);
    }

    let mut nodes: Vec<i32> = graph.keys().map(|x| *x).collect();
    nodes.sort_by(|a, b| {
        graph
            .get(a)
            .unwrap()
            .len()
            .partial_cmp(&(graph.get(b).unwrap().len()))
            .unwrap()
    });

    let mut tree: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut root = -1;
    for i in 0..nodes.len() {
        let mut p = i + 1;
        let leaf = nodes.get(i).unwrap();
        while p < nodes.len() && !graph.get(nodes.get(p).unwrap()).unwrap().contains(leaf) {
            p += 1;
        }
        if p < nodes.len() {
            (*tree.entry(*nodes.get(p).unwrap()).or_insert(Vec::new())).push(*leaf);
            if graph.get(nodes.get(p).unwrap()).unwrap().len() == graph.get(leaf).unwrap().len() {
                result = 2;
            }
        } else {
            if root == -1 {
                root = *leaf;
            } else {
                return 0;
            }
        }
    }

    fn solve_by_dfs(
        root: i32,
        depth: i32,
        result: i32,
        tree: RefCell<HashMap<i32, Vec<i32>>>,
        graph: &HashMap<i32, HashSet<i32>>,
        visited: &mut HashSet<i32>,
    ) -> (i32, i32) {
        if result == 0 {
            return (-1, 0);
        }
        if visited.contains(&root) {
            return (-1, 0);
        }

        visited.insert(root);

        let mut descendants_num = 0;
        tree.borrow_mut().entry(root).or_insert(Vec::new());
        if let Some(nodes) = tree.borrow().get(&root) {
            nodes.iter().for_each(|node| {
                descendants_num += solve_by_dfs(
                    *node,
                    depth + 1,
                    result,
                    RefCell::clone(&tree),
                    graph,
                    visited,
                )
                .0
            });
        }

        if descendants_num + depth != graph.get(&root).unwrap().len() as i32 {
            return (-1, 0);
        }

        return (descendants_num + 1, result);
    }

    solve_by_dfs(
        root,
        0,
        result,
        RefCell::new(tree),
        &graph,
        &mut HashSet::new(),
    )
    .1
}

/// 20
pub fn decode(encoded: Vec<i32>, first: i32) -> Vec<i32> {
    let (mut ret, mut p) = (Vec::new(), first);
    ret.push(first);
    for e in encoded {
        let n = e ^ p;
        ret.push(n);
        p = n;
    }
    ret
}

/// 21
pub fn swap_nodes(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut n = 0;
    let mut ptr = head.as_ref();
    while let Some(node) = ptr {
        n += 1;
        ptr = node.next.as_ref();
    }

    let swap1 = k.min(n + 1 - k);
    let swap2 = k.max(n + 1 - k);
    if swap1 == swap2 {
        return head;
    }

    let mut head = head;
    let mut ptr = head.as_mut();
    let mut swap1_val = 0;
    let mut swap2_val = 0;

    let mut i = 1;
    while let Some(node) = ptr {
        if i == swap1 {
            swap1_val = node.val;
        }
        if i == swap2 {
            swap2_val = node.val;
        }
        i += 1;
        ptr = node.next.as_mut();
    }

    let mut ptr = head.as_mut();
    let mut i = 1;
    while let Some(node) = ptr {
        if i == swap1 {
            node.val = swap2_val;
        }
        if i == swap2 {
            node.val = swap1_val;
        }
        i += 1;
        ptr = node.next.as_mut();
    }

    head
}

/// 22
pub fn minimum_hamming_distance(
    source: Vec<i32>,
    target: Vec<i32>,
    allowed_swaps: Vec<Vec<i32>>,
) -> i32 {
    use std::collections::HashMap;
    let n = source.len();
    let mut unfd = UnionFind::with_capacity(n);
    allowed_swaps
        .into_iter()
        .for_each(|v| unfd.union(v[0] as usize, v[1] as usize));
    let mut indice = HashMap::new();

    source
        .iter()
        .enumerate()
        .for_each(|(i, &v)| indice.entry(v).or_insert(Vec::new()).push(i));
    let mut visit = vec![true; n];
    let mut ans = 0;

    'outer: for i in 0..n {
        if let Some(other) = indice.get(&target[i]) {
            for &j in other.iter() {
                if visit[j] && unfd.is_connected(i, j) {
                    visit[j] = false;
                    continue 'outer;
                }
            }
            ans += 1;
        } else {
            ans += 1;
        }
    }
    ans
}

struct UnionFind {
    parents: Vec<usize>,
    size: Vec<usize>,
    count: usize,
}
impl UnionFind {
    fn with_capacity(n: usize) -> Self {
        Self {
            parents: (0..n).collect(),
            size: vec![1; n],
            count: n,
        }
    }
    fn find(&mut self, node: usize) -> usize {
        while self.parents[node] != self.parents[self.parents[node]] {
            self.parents[node] = self.parents[self.parents[node]];
        }
        self.parents[node]
    }
    fn union(&mut self, node_a: usize, node_b: usize) {
        let p_a = self.find(node_a);
        let p_b = self.find(node_b);
        if p_a != p_b {
            self.count -= 1;
            if self.size[p_a] >= self.size[p_b] {
                self.parents[p_b] = p_a;
                self.size[p_a] += self.size[p_b];
            } else {
                self.parents[p_a] = p_b;
                self.size[p_b] += self.size[p_a];
            }
        }
    }
    fn is_connected(&mut self, node_a: usize, node_b: usize) -> bool {
        self.find(node_a) == self.find(node_b)
    }
}

/// 23
pub fn minimum_time_required(mut jobs: Vec<i32>, k: i32) -> i32 {
    fn check(jobs: &Vec<i32>, limit: i32, k: usize) -> bool {
        let mut workloads = vec![0; k];

        return backtrack(jobs, &mut workloads, k, limit, 0);
    }

    fn backtrack(
        jobs: &Vec<i32>,
        workloads: &mut Vec<i32>,
        k: usize,
        limit: i32,
        index: usize,
    ) -> bool {
        if index >= jobs.len() {
            return true;
        }

        let job = jobs[index];

        for i in 0..k {
            if workloads[i] + job <= limit {
                workloads[i] += job;

                if backtrack(jobs, workloads, k, limit, index + 1) {
                    return true;
                }

                workloads[i] -= job;

                if workloads[i] == 0 {
                    break;
                }
            }
        }

        false
    }
    jobs.sort_by(|a, b| b.cmp(a));

    let mut l = jobs[0];
    let mut r = jobs.iter().sum();
    let k = k as usize;

    while l < r {
        let m = (l + r) >> 1;

        if check(&jobs, m, k) {
            r = m;
        } else {
            l = m + 1;
        }
    }

    l
}

/// 25
pub fn count_good_rectangles(rectangles: Vec<Vec<i32>>) -> i32 {
    rectangles
        .iter()
        .fold((0, 0), |a, c| {
            let cur = c[0].min(c[1]);
            if cur > a.0 {
                (cur, 1)
            } else if cur == a.0 {
                (a.0, a.1 + 1)
            } else {
                (a.0, a.1)
            }
        })
        .1
}

/// 26
pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
    let mut cnt = HashMap::new();
    for i in 0..nums.len() {
        for j in i + 1..nums.len() {
            *cnt.entry(nums[i] * nums[j]).or_insert(0) += 1;
        }
    }
    cnt.values().into_iter().fold(0, |a, c| a + c * (c - 1) * 4)
}

/// 27
pub fn largest_submatrix(matrix: Vec<Vec<i32>>) -> i32 {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut matrix = matrix;
    let mut max_area = 0;

    for i in 1..m {
        for j in 0..n {
            if matrix[i][j] == 1 {
                matrix[i][j] += matrix[i - 1][j];
            }
        }
    }

    for i in 0..m {
        matrix[i].sort_by(|a, b| b.cmp(a));
        for j in 0..n {
            let area = (j as i32 + 1) * matrix[i][j];
            if area > max_area {
                max_area = area;
            }
        }
    }

    max_area
}

/// 28
pub fn can_mouse_win(grid: Vec<String>, cat_jump: i32, mouse_jump: i32) -> bool {
    static MOUSE_TURN: usize = 0;
    static CAT_TURN: usize = 1;

    fn build_graph(jump: i32, grid_arr: &Vec<Vec<char>>) -> Vec<Vec<i32>> {
        let (m, n) = (grid_arr.len(), grid_arr[0].len());
        let mut graph = vec![Vec::new(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut list = Vec::new();
                if grid_arr[i][j] == '#' {
                    continue;
                }
                list.push((i * n + j) as i32);
                for dir in [[-1, 0], [1, 0], [0, -1], [0, 1]] {
                    for k in 1..=jump {
                        let (x, y) = (i as i32 + dir[0] * k, j as i32 + dir[1] * k);
                        if x < 0
                            || x as usize >= m
                            || y < 0
                            || y as usize >= n
                            || grid_arr[x as usize][y as usize] == '#'
                        {
                            break;
                        }
                        list.push(x * n as i32 + y);
                    }
                }
                graph[i * n + j] = list;
            }
        }
        graph
    }

    fn dfs<'a>(
        p1: usize,
        p2: usize,
        food_pos: i32,
        mut turn: usize,
        memo: &'a RefCell<Vec<Vec<Vec<i32>>>>,
        graph: &Vec<Vec<Vec<i32>>>,
    ) -> &'a RefCell<Vec<Vec<Vec<i32>>>> {
        if p1 == p2 {
            return &memo;
        }
        let dst = if turn == 0 { p2 } else { p1 };
        if dst == food_pos as usize {
            return &memo;
        }
        if memo.borrow()[p1][p2][turn as usize] < 0 {
            return &memo;
        }
        memo.borrow_mut()[p1][p2][turn as usize] = -1;
        turn ^= 1;
        for w in &graph[turn as usize][p2] {
            if turn == MOUSE_TURN {
                dfs(*w as usize, p1, food_pos, turn, memo, graph);
            } else {
                memo.borrow_mut()[*w as usize][p1][turn as usize] += 1;
                if memo.borrow()[*w as usize][p1][turn as usize] as usize
                    == graph[turn as usize][*w as usize].len()
                {
                    dfs(*w as usize, p1, food_pos, turn, memo, graph);
                }
            }
        }
        &memo
    }

    let grid_arr = grid
        .iter()
        .map(|g| g.chars().collect::<Vec<char>>())
        .collect::<Vec<_>>();
    let (m, n) = (grid.len(), grid[0].len());
    let (mut mouse_pos, mut cat_pos, mut food_pos) = (0, 0, 0);

    for i in 0..m {
        for j in 0..n {
            match grid_arr[i][j] {
                'F' => food_pos = i * n + j,
                'C' => cat_pos = i * n + j,
                'M' => mouse_pos = i * n + j,
                _ => {}
            }
        }
    }
    let mut graph: Vec<Vec<Vec<i32>>> = vec![Vec::new(); 2];
    graph[0] = build_graph(mouse_jump, &grid_arr);
    graph[1] = build_graph(cat_jump, &grid_arr);
    let memo = RefCell::new(vec![vec![vec![0; 2]; m * n]; m * n]);
    for i in 0..m {
        for j in 0..n {
            let ch = grid_arr[i][j];
            if ch == '#' || ch == 'F' {
                continue;
            }
            let ret = dfs(
                i * n + j,
                food_pos,
                food_pos as i32,
                CAT_TURN as usize,
                &memo,
                &graph,
            )
            .borrow()[mouse_pos][cat_pos][MOUSE_TURN]
                < 0;
            if ret {
                return true;
            }
        }
    }
    false
}

/// 32
pub fn largest_altitude(gain: Vec<i32>) -> i32 {
    gain.iter()
        .fold((0, 0), |a, c| (a.0 + c, a.1.max(a.0 + c)))
        .1
}

/// 33
pub fn minimum_teachings(_: i32, languages: Vec<Vec<i32>>, friendships: Vec<Vec<i32>>) -> i32 {
    let mut cncon = HashSet::new();
    for friendship in friendships {
        let mut mp = HashSet::new();
        let mut conm = false;
        for &lan in &languages[friendship[0] as usize - 1] {
            mp.insert(lan);
        }
        for &lan in &languages[friendship[1] as usize - 1] {
            if mp.contains(&lan) {
                conm = true;
                break;
            }
        }

        if !conm {
            cncon.insert(friendship[0] - 1);
            cncon.insert(friendship[1] - 1);
        }
    }

    let mut max_cnt = 0;
    let mut cnt = HashMap::new();
    for &person in &cncon {
        for &lan in &languages[person as usize] {
            *cnt.entry(lan).or_insert(0) += 1;
            max_cnt = max_cnt.max(*cnt.get(&lan).unwrap());
        }
    }

    cncon.len() as i32 - max_cnt
}

/// 34
pub fn decode_ii(mut encoded: Vec<i32>) -> Vec<i32> {
    let n = encoded.len() + 1;
    let mut s = encoded[1..]
        .chunks(2)
        .fold((2..=n as i32).fold(1, |s, x| s ^ x), |s, x| s ^ x[0]);
    encoded.push(0);
    encoded.iter_mut().for_each(|x| {
        let t = *x;
        *x = s;
        s ^= t
    });
    encoded
}

/// 35
pub fn ways_to_fill_array(queries: Vec<Vec<i32>>) -> Vec<i32> {
    const MOD: i64 = 1_000_000_007;
    const MAXN: usize = 10_014;
    const MAXM: usize = 14;

    let mut comb = vec![vec![0; MAXM]; MAXN];
    let mut ans = Vec::new();

    comb[0][0] = 1;
    for i in 1..MAXN {
        comb[i][0] = 1;
        for j in 1..=i.min(MAXM - 1) {
            comb[i][j] = (comb[i - 1][j - 1] + comb[i - 1][j]) % MOD;
        }
    }

    for q in queries {
        let n = q[0] as usize;
        let mut k = q[1] as i64;
        let mut tot = 1;

        let mut i = 2;
        while i * i <= k {
            if k % i == 0 {
                let mut cnt = 0;
                while k % i == 0 {
                    k /= i;
                    cnt += 1;
                }
                tot = (tot * comb[n + cnt - 1][cnt]) % MOD;
            }
            i += 1;
        }
        if k > 1 {
            tot = (tot * n as i64) % MOD;
        }
        ans.push(tot as i32);
    }
    ans
}

/// 36
pub fn maximum_time(time: String) -> String {
    let times = time.as_bytes();
    let mut ans = String::new();

    if times[0] == b'?' {
        ans.push(if b'4' <= times[1] && times[1] <= b'9' {
            '1'
        } else {
            '2'
        });
    } else {
        ans.push(times[0] as char);
    }

    if times[1] == b'?' {
        ans.push(if times[0] == b'2' || times[0] == b'?' {
            '3'
        } else {
            '9'
        });
    } else {
        ans.push(times[1] as char);
    }

    ans.push(':');

    if times[3] == b'?' {
        ans.push('5');
    } else {
        ans.push(times[3] as char);
    }

    if times[4] == b'?' {
        ans.push('9');
    } else {
        ans.push(times[4] as char);
    }

    ans
}

/// 37
pub fn min_characters(a: String, b: String) -> i32 {
    let mut ac = vec![0; 26];
    for c in a.bytes() {
        let c = c as usize - b'a' as usize;
        ac[c] += 1;
    }
    let mut bc = vec![0; 26];
    for c in b.bytes() {
        let c = c as usize - b'a' as usize;
        bc[c] += 1;
    }
    let mut ans = a.len() + b.len() - (0..ac.len()).map(|i| ac[i] + bc[i]).max().unwrap();
    let (sa, sb) = (a.len(), b.len());
    let mut a = 0;
    let mut b = 0;
    for i in 0..25 {
        a += ac[i];
        b += bc[i];
        ans = ans.min(a + sb - b).min(b + sa - a);
    }

    ans as _
}

/// 38
pub fn kth_largest_value(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
    let (height, width) = (matrix.len(), matrix[0].len());
    let (mut prefix, mut all) = (vec![vec![0; width + 1]; height + 1], vec![]);
    for r in 0..height {
        for c in 0..width {
            prefix[r + 1][c + 1] =
                prefix[r][c + 1] ^ prefix[r + 1][c] ^ prefix[r][c] ^ matrix[r][c];
            all.push(prefix[r + 1][c + 1]);
        }
    }
    all.sort();
    all[all.len() - k as usize]
}

/// 39
pub fn minimum_boxes(mut n: i32) -> i32 {
    let mut base = (6_f64 * n as f64).cbrt() as i64;
    if base * (base + 1) * (base + 2) / 6_i64 > n as i64 {
        base -= 1;
    }
    let block_base = base * (base + 1) / 2;
    n -= (base * (base + 1) * (base + 2) / 6) as i32;
    let extra = (2_f64 * n as f64).sqrt() as i64;
    if extra * (extra + 1) / 2 >= n as i64 {
        (block_base + extra) as i32
    } else {
        (block_base + extra) as i32 + 1
    }
}

/// 42
pub fn count_balls(low_limit: i32, high_limit: i32) -> i32 {
    let mut cnt = HashMap::new();
    for mut i in low_limit..=high_limit {
        let mut key = 0;
        while i > 0 {
            key += i % 10;
            i = i / 10;
        }
        *cnt.entry(key).or_insert(0) += 1;
    }
    *cnt.values().into_iter().max().unwrap()
}

/// 43
pub fn restore_array(adjacent_pairs: Vec<Vec<i32>>) -> Vec<i32> {
    let (mut cnt, mut ans) = (HashMap::new(), vec![0; adjacent_pairs.len() + 1]);
    adjacent_pairs.iter().for_each(|p| {
        cnt.entry(p[0]).or_insert(vec![]).push(p[1]);
        cnt.entry(p[1]).or_insert(vec![]).push(p[0]);
    });
    for (k, v) in cnt.iter() {
        if v.len() == 1 {
            ans[0] = *k;
            ans[1] = v[0];
            break;
        }
    }
    for i in 2..=adjacent_pairs.len() {
        if let Some(v) = cnt.get(&ans[i - 1]) {
            ans[i] = if v[0] == ans[i - 2] { v[1] } else { v[0] }
        }
    }
    ans
}

/// 44
pub fn can_eat(candies_count: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<bool> {
    let n = candies_count.len();
    let mut candies_prefix_sum = vec![0_i64; n + 1];
    for i in 0..n {
        candies_prefix_sum[i + 1] = candies_prefix_sum[i] + candies_count[i] as i64;
    }
    let mut answer = vec![false; queries.len()];
    for (i, query) in queries.into_iter().enumerate() {
        let f_type = query[0] as usize;
        let f_day = query[1] as i64;
        let d_cap = query[2] as i64;
        if f_day >= candies_prefix_sum[f_type + 1] {
            continue;
        }
        let min = candies_prefix_sum[f_type] + 1;
        if d_cap * (f_day + 1) >= min {
            answer[i] = true;
        }
    }
    answer
}

/// 45
pub fn check_partitioning(s: String) -> bool {
    let chars = s.chars().collect::<Vec<char>>();
    let length = chars.len();
    let mut is_pal = vec![vec![false; length]; length];
    for l in 1..length {
        for s in 0..=length - l {
            let e = s + l - 1;
            if l == 1 {
                is_pal[s][e] = true;
            } else if l == 2 {
                is_pal[s][e] = chars[s] == chars[e];
            } else {
                is_pal[s][e] = chars[s] == chars[e] && is_pal[s + 1][e - 1];
            }
        }
    }
    for l in 1..=length - 2 {
        if !is_pal[0][l - 1] {
            continue;
        }
        for r in l..=length - 2 {
            if is_pal[l][r] && is_pal[r + 1][length - 1] {
                return true;
            }
        }
    }
    false
}

/// 48
pub fn sum_of_unique(nums: Vec<i32>) -> i32 {
    nums.iter()
        .fold((HashSet::new(), HashSet::new()), |mut a, c| {
            if a.1.contains(c) {
                a
            } else if a.0.contains(c) {
                a.0.remove(c);
                a.1.insert(c);
                a
            } else {
                a.0.insert(c);
                a
            }
        })
        .0
        .into_iter()
        .sum()
}

/// 49
pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
    let (mut dp, mut max) = (vec![(i32::MIN, i32::MAX); nums.len()], i32::MIN);
    dp[0] = (nums[0], nums[0]);
    max = max.max(nums[0].abs());
    for i in 1..nums.len() {
        if dp[i - 1].0 < 0 {
            dp[i].0 = nums[i];
            dp[i].1 = dp[i - 1].1 + nums[i];
        } else if dp[i - 1].1 < 0 {
            dp[i].0 = dp[i - 1].0 + nums[i];
            dp[i].1 = dp[i - 1].1 + nums[i];
        } else {
            dp[i].0 = dp[i - 1].0 + nums[i];
            dp[i].1 = nums[i];
        }
        max = max.max(dp[i].0.abs().max(dp[i].1.abs()));
    }
    max
}

/// 50
pub fn minimum_length(s: String) -> i32 {
    let (mut l, mut r, s_arr) = (0, s.len() - 1, s.as_bytes());
    while l < r && s_arr[l] == s_arr[r] {
        let target = s_arr[l];
        while l <= r && s_arr[l] == target {
            l += 1;
        }
        while l <= r && s_arr[r] == target {
            r -= 1;
        }
    }
    (r - l + 1) as i32
}

/// 51
pub fn max_value(mut events: Vec<Vec<i32>>, k: i32) -> i32 {
    events.sort_by_key(|e| e[1]);
    let n = events.len();
    let k = k as usize;
    let mut dp = vec![vec![0; k + 1]; n + 1];

    for i in 0..n {
        let p = events.partition_point(|x| x[1] < events[i][0]);
        for j in 1..=k {
            dp[i + 1][j] = dp[i][j].max(dp[p][j - 1] + events[i][2]);
        }
    }

    dp[n][k]
}

/// 52
pub fn check(nums: Vec<i32>) -> bool {
    let mut reverse = 0;
    for i in 0..nums.len() {
        if nums[i] > nums[(i + 1) % nums.len()] {
            reverse += 1;
            if reverse > 1 {
                return false;
            }
        }
    }
    true
}

/// 53
pub fn maximum_score(a: i32, b: i32, c: i32) -> i32 {
    ((a + b + c) / 2).min(a + b + c - a.max(b).max(c))
}

/// 54
pub fn largest_merge(word1: String, word2: String) -> String {
    let (mut i, mut j, mut ret, word1, word2) = (
        0,
        0,
        String::with_capacity(word1.len() + word2.len()),
        word1.into_bytes(),
        word2.into_bytes(),
    );
    while i < word1.len() || j < word2.len() {
        if word1[i..] > word2[j..] {
            ret.push(word1[i] as char);
            i += 1
        } else {
            ret.push(word2[j] as char);
            j += 1
        }
    }
    ret
}

/// 55
pub fn min_abs_difference(nums: Vec<i32>, goal: i32) -> i32 {
    fn get_sums(nums: &Vec<i32>) -> (BTreeSet<i32>, BTreeSet<i32>) {
        let n = nums.len();

        let left_len = n / 2;
        let left_total_status = (1 << left_len) as usize;

        let mut left_sums = vec![0; left_total_status];
        for i in 1..left_total_status {
            for j in 0..left_len {
                if i & (1 << j) == 0 {
                    continue;
                }
                left_sums[i] = left_sums[i - (1 << j)] + nums[j];
                break;
            }
        }

        let right_len = n - left_len;
        let right_total_status = (1 << right_len) as usize;

        let mut right_sums = vec![0; right_total_status];
        for i in 1..right_total_status {
            for j in 0..right_len {
                if i & (1 << j) == 0 {
                    continue;
                }
                right_sums[i] = right_sums[i - (1 << j)] + nums[j + left_len];
                break;
            }
        }

        (
            left_sums.into_iter().collect::<BTreeSet<i32>>(),
            right_sums.into_iter().collect::<BTreeSet<i32>>(),
        )
    }

    fn get_min_diff(left_sums: &BTreeSet<i32>, right_sums: &BTreeSet<i32>, target: i32) -> i32 {
        let mut min_diff = i32::MAX;

        for &val1 in left_sums.iter() {
            let val2 = target - val1;
            if let Some(&val2) = right_sums.range(val2..).next() {
                min_diff = i32::min(min_diff, (target - (val1 + val2)).abs());
            }
            if let Some(&val2) = right_sums.range(..val2).next_back() {
                min_diff = i32::min(min_diff, (target - (val1 + val2)).abs());
            }
        }

        min_diff
    }
    let (left_sums, right_sums) = get_sums(&nums);
    get_min_diff(&left_sums, &right_sums, goal)
}

/// 58
pub fn min_operations_ii(s: String) -> i32 {
    let (chars, mut cnt) = (s.chars().collect::<Vec<char>>(), 0);
    for i in 0..chars.len() {
        if i % 2 == 0 && chars[i] != '0' {
            cnt += 1
        }
        if i % 2 == 1 && chars[i] != '1' {
            cnt += 1
        }
    }
    cnt.min(chars.len() - cnt) as i32
}

/// 59
pub fn count_homogenous(s: String) -> i32 {
    s.as_bytes()
        .windows(2)
        .fold((1, 1), |(cnt, ret), ch| {
            if ch[0] == ch[1] {
                (cnt + 1, (ret + cnt + 1) % 1_000_000_007)
            } else {
                (1, (ret + 1) % 1_000_000_007)
            }
        })
        .1
}

/// 60
pub fn minimum_size(nums: Vec<i32>, max_operations: i32) -> i32 {
    let (mut left, mut right, mut ans) = (1, *nums.iter().max().unwrap(), 0);
    while left <= right {
        let (t, mut op) = ((left + right) / 2, 0 as i64);
        for n in nums.iter() {
            op += ((n - 1) / t) as i64;
        }
        if op <= max_operations as i64 {
            ans = t;
            right = t - 1;
        } else {
            left = t + 1;
        }
    }
    ans
}

/// 61
pub fn min_trio_degree(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    let n = n as usize;
    let mut g = vec![vec![false; n]; n];
    let mut dgr = vec![0; n];
    for edge in edges {
        let (u, v) = (edge[0] as usize - 1, edge[1] as usize - 1);
        g[u][v] = true;
        g[v][u] = true;
        dgr[u] += 1;
        dgr[v] += 1;
    }
    let mut res = i32::MAX;
    (0..n).for_each(|i| {
        (i + 1..n).for_each(|j| {
            if !g[i][j] {
                return;
            }
            (j + 1..n).for_each(|k| {
                if g[i][k] && g[j][k] {
                    res = res.min(dgr[i] + dgr[j] + dgr[k] - 6);
                }
            })
        })
    });
    if res == i32::MAX { -1 } else { res }
}

/// 63
pub fn longest_nice_substring(s: String) -> String {
    let cache: HashSet<char> = s.chars().collect();

    for (i, ch) in s.chars().enumerate() {
        if cache.contains(&ch.to_ascii_uppercase()) && cache.contains(&ch.to_ascii_lowercase()) {
            continue;
        }
        let (s1, s2) = (
            longest_nice_substring(s[0..i].to_string()),
            longest_nice_substring(s[i + 1..].to_string()),
        );
        return if s1.len() >= s2.len() { s1 } else { s2 };
    }
    s
}

/// 64
pub fn can_choose(groups: Vec<Vec<i32>>, nums: Vec<i32>) -> bool {
    let (mut i, mut j, m, n) = (0, 0, groups.len(), nums.len());
    while j < n {
        let (mut p, mut q) = (j, 0);
        while p < n && q < groups[i].len() && groups[i][q] == nums[p] {
            p += 1;
            q += 1;
        }
        if groups[i].len() == q {
            i += 1;
            j = p - 1;
        }
        if i == m {
            return true;
        }
        j += 1;
    }
    i == m
}

/// 65
pub fn highest_peak(is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (rows, cols) = (is_water.len(), is_water[0].len());
    let mut visited = vec![vec![false; cols]; rows];
    let mut q = VecDeque::new();
    for i in 0..rows {
        for j in 0..cols {
            if is_water[i][j] == 1 {
                visited[i][j] = true;
                q.push_back((i, j));
            }
        }
    }
    let mut ans = vec![vec![0; cols]; rows];
    let mut high = 0;
    while !q.is_empty() {
        high += 1;
        for _ in 0..q.len() {
            let (i, j) = q.pop_front().unwrap();
            for (x, y) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)] {
                if x < rows && y < cols && !visited[x][y] {
                    ans[x][y] = high;
                    visited[x][y] = true;
                    q.push_back((x, y));
                }
            }
        }
    }
    ans
}

/// 66
pub fn get_coprimes(nums: Vec<i32>, edges: Vec<Vec<i32>>) -> Vec<i32> {
    let n = nums.len();
    let mut gcds = vec![vec![]; 51];
    let mut tmp = vec![vec![]; 51];
    let mut ans = vec![-1; n];
    let mut dep = vec![-1; n];
    let mut g = vec![vec![]; n];

    fn gcd(a: i32, b: i32) -> i32 {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a.abs()
    }

    fn dfs(
        x: usize,
        depth: i32,
        dep: &mut [i32],
        ans: &mut [i32],
        nums: &Vec<i32>,
        tmp: &mut Vec<Vec<usize>>,
        gcds: &Vec<Vec<i32>>,
        g: &Vec<Vec<usize>>,
    ) {
        dep[x] = depth;
        for &val in &gcds[nums[x] as usize] {
            if tmp[val as usize].is_empty() {
                continue;
            }
            let las = *tmp[val as usize].last().unwrap();
            if ans[x] == -1 || dep[las] > dep[ans[x] as usize] {
                ans[x] = las as i32;
            }
        }
        tmp[nums[x] as usize].push(x);
        for &val in &g[x] {
            if dep[val] == -1 {
                dfs(val, depth + 1, dep, ans, nums, tmp, gcds, g);
            }
        }
        tmp[nums[x] as usize].pop();
    }

    for i in 1..=50 {
        for j in 1..=50 {
            if gcd(i, j) == 1 {
                gcds[i as usize].push(j);
            }
        }
    }
    for edge in &edges {
        let x = edge[0] as usize;
        let y = edge[1] as usize;
        g[x].push(y);
        g[y].push(x);
    }
    dfs(0, 1, &mut dep, &mut ans, &nums, &mut tmp, &gcds, &g);
    ans
}

/// 68
pub fn merge_alternately(word1: String, word2: String) -> String {
    let (char_1, char_2) = (
        word1.chars().collect::<Vec<char>>(),
        word2.chars().collect::<Vec<char>>(),
    );
    let mut ans = "".to_string();
    for i in 0..char_1.len().min(char_2.len()) {
        ans.push(char_1[i]);
        ans.push(char_2[i]);
    }
    if char_1.len() > char_2.len() {
        for i in 0..char_1.len() - char_2.len() {
            ans.push(char_1[char_2.len() + i]);
        }
    }
    if char_1.len() < char_2.len() {
        for i in 0..char_2.len() - char_1.len() {
            ans.push(char_2[char_1.len() + i]);
        }
    }
    ans
}

/// 69
pub fn min_operations_iii(boxes: String) -> Vec<i32> {
    let chars = boxes.chars().collect::<Vec<char>>();
    let mut ans = vec![0; chars.len()];
    for i in 0..chars.len() {
        for j in 0..chars.len() {
            if chars[j] == '1' {
                ans[i] += (i as i32 - j as i32).abs()
            }
        }
    }
    ans
}

/// 70
pub fn maximum_score_ii(nums: Vec<i32>, multipliers: Vec<i32>) -> i32 {
    let (n, m) = (nums.len(), multipliers.len());
    let mut dp = vec![vec![i32::MIN; m + 1]; m + 1];
    dp[0][0] = 0;
    for i in 0..=m {
        for j in 0..=m - i {
            if i > 0 {
                dp[i][j] = dp[i][j].max(dp[i - 1][j] + multipliers[i + j - 1] * nums[i - 1]);
            }
            if j > 0 {
                dp[i][j] = dp[i][j].max(dp[i][j - 1] + multipliers[i + j - 1] * nums[n - j]);
            }
        }
    }
    let mut ans = i32::MIN;
    for i in 0..=m {
        ans = ans.max(dp[i][m - i]);
    }
    ans
}

/// 71
pub fn longest_palindrome(word1: String, word2: String) -> i32 {
    let mut chars = word1.chars().collect::<Vec<char>>();
    let length_1 = chars.len();
    chars.append(&mut word2.chars().collect::<Vec<char>>());
    let total_length = chars.len();
    let mut dp = vec![vec![0; total_length]; total_length];
    let mut ans = 0;
    for i in (0..total_length).rev() {
        for j in i..total_length {
            if i == j {
                dp[i][j] = 1;
            } else {
                if chars[i] == chars[j] {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                    if i < length_1 && length_1 <= j {
                        ans = ans.max(dp[i][j]);
                    }
                } else {
                    dp[i][j] = dp[i][j].max(dp[i + 1][j].max(dp[i][j - 1]));
                }
            }
        }
    }
    ans
}

/// 73
pub fn count_matches(items: Vec<Vec<String>>, rule_key: String, rule_value: String) -> i32 {
    items.iter().fold(0, |a, c| match rule_key.as_str() {
        "type" => {
            if c[0] == rule_value {
                a + 1
            } else {
                a
            }
        }
        "color" => {
            if c[1] == rule_value {
                a + 1
            } else {
                a
            }
        }
        "name" => {
            if c[2] == rule_value {
                a + 1
            } else {
                a
            }
        }
        _ => a,
    })
}

/// 74
pub fn closest_cost(base_costs: Vec<i32>, topping_costs: Vec<i32>, target: i32) -> i32 {
    static mut RET: i32 = i32::MAX;
    static mut DIFF: i32 = i32::MAX;

    fn back_track(topping_costs: &Vec<i32>, cost: i32, target: i32, curr_idx: usize) {
        let cost_diff = (target - cost).abs();
        unsafe {
            if cost_diff == DIFF && cost < RET || cost_diff < DIFF {
                DIFF = cost_diff;
                RET = cost;
            }
        }
        if curr_idx >= topping_costs.len() || cost > target {
            return;
        }
        back_track(topping_costs, cost, target, curr_idx + 1);
        back_track(
            topping_costs,
            cost + topping_costs[curr_idx],
            target,
            curr_idx + 1,
        );
        back_track(
            topping_costs,
            cost + topping_costs[curr_idx] * 2,
            target,
            curr_idx + 1,
        );
    }
    unsafe {
        RET = i32::MAX;
        DIFF = i32::MAX;
    }
    for base in base_costs {
        back_track(&topping_costs, base, target, 0);
    }
    unsafe { RET }
}

/// 75
pub fn min_operations_iv(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let s1: i32 = nums1.iter().sum();
    let s2: i32 = nums2.iter().sum();
    let (l, r, t) = if s1 < s2 {
        (nums1, nums2, s2 - s1)
    } else {
        (nums2, nums1, s1 - s2)
    };
    if t == 0 {
        return 0;
    }
    let mut q: Vec<_> = l
        .into_iter()
        .map(|i| 6 - i)
        .chain(r.into_iter().map(|i| i - 1))
        .collect();
    q.sort_unstable();
    q.into_iter()
        .rev()
        .scan(t, |t, n| {
            *t -= n;
            Some(*t)
        })
        .zip(1..)
        .find(|v| v.0 <= 0)
        .map_or(-1, |v| v.1)
}

/// 79
pub fn nearest_valid_point(x: i32, y: i32, points: Vec<Vec<i32>>) -> i32 {
    points
        .iter()
        .enumerate()
        .fold((-1, i32::MAX), |a, (i, v)| {
            if v[0] == x || v[1] == y {
                let dis = (v[0] - x).abs() + (v[1] - y).abs();
                if dis < a.1 { (i as i32, dis) } else { a }
            } else {
                a
            }
        })
        .0
}

/// 80
pub fn check_powers_of_three(mut n: i32) -> bool {
    while n > 0 {
        if n % 3 == 2 {
            return false;
        }
        n = n / 3;
    }
    true
}
