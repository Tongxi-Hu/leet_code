use std::collections::{HashSet, VecDeque};

/// p801
pub fn min_swap(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let x = (1..nums1.len()).fold((1, 0), |(a, b), i| {
        if nums1[i - 1] >= nums2[i] || nums2[i - 1] >= nums1[i] {
            (a + 1, b)
        } else if nums1[i - 1] >= nums1[i] || nums2[i - 1] >= nums2[i] {
            (b + 1, a)
        } else {
            (a.min(b) + 1, a.min(b))
        }
    });
    x.0.min(x.1)
}

/// p802
pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
    let n = graph.len();
    let mut out_degree = vec![0; n];
    let mut in_graph = vec![Vec::new(); n];
    let mut queue = VecDeque::new();
    let mut result = vec![];

    for (i, points) in graph.iter().enumerate() {
        points
            .iter()
            .filter(|&&next_node| next_node != i as i32)
            .for_each(|&next_node| {
                in_graph[next_node as usize].push(i);
            });
        out_degree[i] = points.len() as i32;

        if out_degree[i] == 0 {
            result.push(i as i32);
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        for &prev_node in in_graph[node].iter() {
            out_degree[prev_node] -= 1;
            if out_degree[prev_node] == 0 {
                result.push(prev_node as i32);
                queue.push_back(prev_node);
            }
        }
    }

    result.sort_unstable();
    result
}

/// p803
pub struct UnionFindSize {
    parent: Vec<usize>,
    size: Vec<i32>,
}

impl UnionFindSize {
    pub fn new(n: usize) -> Self {
        UnionFindSize {
            parent: (0..=n).collect(),
            size: [1].repeat(n + 1),
        }
    }

    pub fn find(&mut self, idx: usize) -> usize {
        if idx != self.parent[idx] {
            self.parent[idx] = self.find(self.parent[idx]);
        }
        self.parent[idx]
    }

    pub fn union(&mut self, idx1: usize, idx2: usize) {
        let (x, y) = (self.find(idx1), self.find(idx2));
        if x == y {
            return;
        }
        if x == self.size.len() - 1 {
            self.parent[y] = x;
            self.size[x] += self.size[y];
        } else {
            self.parent[x] = y;
            self.size[y] += self.size[x];
        }
    }

    pub fn get_size(&self, idx: usize) -> i32 {
        self.size[idx]
    }
}

pub fn hit_bricks(grid: Vec<Vec<i32>>, hits: Vec<Vec<i32>>) -> Vec<i32> {
    let (rows, cols) = (grid.len(), grid[0].len());
    let root = rows * cols;
    let mut flashback = grid.clone();

    hits.iter()
        .for_each(|h| flashback[h[0] as usize][h[1] as usize] = 0);

    let mut uf = UnionFindSize::new(rows * cols);
    for j in 0..cols {
        if flashback[0][j] == 1 {
            uf.union(j, root);
        }
    }
    for i in 1..rows {
        for j in 0..cols {
            if flashback[i][j] == 0 {
                continue;
            }
            if flashback[i - 1][j] == 1 {
                uf.union(i * cols + j, (i - 1) * cols + j);
            }
            if j > 0 && flashback[i][j - 1] == 1 {
                uf.union(i * cols + j, i * cols + j - 1);
            }
        }
    }

    let mut counts = vec![uf.get_size(root)];
    for hit in hits.iter().rev() {
        let (i, j) = (hit[0] as usize, hit[1] as usize);
        if grid[i][j] == 0 {
            counts.push(uf.get_size(root));
            continue;
        }
        flashback[i][j] = 1;
        for (x, y) in [(i, j + 1), (i, j - 1), (i + 1, j), (i - 1, j)] {
            if x < rows && y < cols && flashback[x][y] == 1 {
                uf.union(i * cols + j, x * cols + y);
            }
        }
        if i == 0 {
            uf.union(j, root);
        }
        counts.push(uf.get_size(root));
    }
    let n = hits.len();
    let mut ans = vec![0; n];
    for i in 0..ans.len() {
        let d = counts[n - i] - counts[n - i - 1] - 1;
        ans[i] = if d < 0 { 0 } else { d };
    }
    ans
}

/// p804
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
    let morse_table = [
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
        "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--",
        "--..",
    ];
    words
        .iter()
        .fold(HashSet::new(), |mut count, word| {
            let mut s = String::new();
            word.chars()
                .for_each(|c| s.push_str(morse_table[((c as u8).abs_diff('a' as u8)) as usize]));
            count.insert(s);
            count
        })
        .len() as i32
}

/// p805
pub fn split_array_same_average(nums: Vec<i32>) -> bool {
    let (sum, n) = (nums.iter().sum::<i32>(), nums.len());
    let mut dp = vec![0; sum as usize + 1];
    dp[0] = 1;
    for num in nums {
        for s in (num..=sum).rev() {
            if dp[(s - num) as usize] > 0 {
                dp[s as usize] |= dp[(s - num) as usize] << 1;
            }
        }
    }
    for i in 1..n {
        if sum * i as i32 % n as i32 != 0 {
            continue;
        }
        let s = sum * i as i32 / n as i32;
        if dp[s as usize] > 0 && (dp[s as usize] & (1 << i as i32)) > 0 {
            return true;
        }
    }
    false
}
