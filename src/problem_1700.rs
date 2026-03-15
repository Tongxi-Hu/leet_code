use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    f64::consts::PI,
    rc::Rc,
};

use crate::common::TreeNode;

/// 01
pub fn maximum_requests(n: i32, mut requests: Vec<Vec<i32>>) -> i32 {
    let mut cache = vec![0; 21];
    fn back_track(n: i32, i: i32, cache: &mut Vec<i32>, requests: &mut Vec<Vec<i32>>) -> i32 {
        if i as usize == requests.len() {
            return if (0..n)
                .into_iter()
                .filter(|j| cache[*j as usize] == 0)
                .count()
                == n as usize
            {
                0
            } else {
                i32::MIN
            };
        }
        cache[requests[i as usize][0] as usize] -= 1;
        cache[requests[i as usize][1] as usize] += 1;
        let take = 1 + back_track(n, i + 1, cache, requests);
        cache[requests[i as usize][0] as usize] += 1;
        cache[requests[i as usize][1] as usize] -= 1;
        return take.max(back_track(n, i + 1, cache, requests));
    }
    back_track(n, 0, &mut cache, &mut requests)
}

/// 03
struct ParkingSystem {
    big: (i32, i32),
    medium: (i32, i32),
    small: (i32, i32),
}

impl ParkingSystem {
    fn new(big: i32, medium: i32, small: i32) -> Self {
        Self {
            big: (big, 0),
            medium: (medium, 0),
            small: (small, 0),
        }
    }

    fn add_car(&mut self, car_type: i32) -> bool {
        match car_type {
            1 => {
                if self.big.1 < self.big.0 {
                    self.big.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            2 => {
                if self.medium.1 < self.medium.0 {
                    self.medium.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            3 => {
                if self.small.1 < self.small.0 {
                    self.small.1 += 1;
                    return true;
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// 04
pub fn alert_names(key_name: Vec<String>, key_time: Vec<String>) -> Vec<String> {
    let mut mp = std::collections::HashMap::new();
    let n = key_name.len();

    for i in 0..n {
        let time = key_time[i][0..2].parse::<i32>().unwrap() * 60
            + key_time[i][3..].parse::<i32>().unwrap();
        mp.entry(&key_name[i]).or_insert(vec![]).push(time);
    }

    let mut ans = vec![];

    for (key, val) in mp.iter_mut() {
        val.sort();
        if val.len() < 3 {
            continue;
        }
        for i in 2..val.len() {
            if val[i] - val[i - 2] <= 60 {
                ans.push(key.to_string());
                break;
            }
        }
    }

    ans.sort();
    ans
}

/// 05
pub fn restore_matrix(mut row_sum: Vec<i32>, mut col_sum: Vec<i32>) -> Vec<Vec<i32>> {
    let (mut r, mut c, height, width) = (0, 0, row_sum.len(), col_sum.len());
    let mut grid = vec![vec![0; width]; height];
    while r < height && c < width {
        let min = row_sum[r].min(col_sum[c]);
        grid[r][c] = min;
        row_sum[r] = row_sum[r] - min;
        col_sum[c] = col_sum[c] - min;
        if row_sum[r] == 0 {
            r = r + 1;
        }
        if col_sum[c] == 0 {
            c = c + 1;
        }
    }
    grid
}

/// 06
pub fn busiest_servers(k: i32, arrival: Vec<i32>, load: Vec<i32>) -> Vec<i32> {
    let mut counts = vec![0; k as usize];
    let mut max_req_handled = 0;
    let mut available_servers = BTreeSet::new();
    let mut queue: BinaryHeap<Reverse<(i32, i32)>> = BinaryHeap::new();
    for i in 0..k {
        available_servers.insert(i);
    }

    for i in 0..arrival.len() {
        let (req_start_time, req_end_time) = (arrival[i], arrival[i] + load[i]);
        while !queue.is_empty() && queue.peek().unwrap().0.0 <= req_start_time {
            available_servers.insert(queue.pop().unwrap().0.1);
        }
        if !available_servers.is_empty() {
            let server = if let Some(s) = available_servers.range(i as i32 % k..).next() {
                *s
            } else {
                *available_servers.range(0..).next().unwrap()
            };
            available_servers.remove(&server);
            queue.push(Reverse((req_end_time, server)));
            counts[server as usize] += 1;
            if counts[server as usize] > max_req_handled {
                max_req_handled = counts[server as usize];
            }
        }
    }
    (0..k)
        .filter(|i| counts[*i as usize] == max_req_handled)
        .collect::<Vec<i32>>()
}

/// 08
pub fn special_array(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    for i in 1..=nums.len() {
        if (nums[nums.len() - i] as usize) >= i
            && (nums.len() - i == 0 || (nums[nums.len() - i - 1] as usize) < i)
        {
            return i as i32;
        }
    }
    return -1;
}

/// 09
pub fn is_even_odd_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    let mut nodes = vec![root.clone()];
    fn bfs(deepth: usize, nodes: &mut Vec<Option<Rc<RefCell<TreeNode>>>>) -> bool {
        let size = nodes.len();
        if size == 0 {
            return true;
        } else {
            let mut values = (0..size).fold(vec![], |mut values, _| {
                let node = nodes.remove(0);
                if let Some(n) = node.as_ref() {
                    values.push(n.borrow().val);
                    nodes.push(n.borrow().left.clone());
                    nodes.push(n.borrow().right.clone());
                }
                values
            });
            if deepth % 2 == 0 {
                if values.is_sorted_by(|a, b| a < b) && values.iter().all(|a| a % 2 == 1) {
                    return bfs(deepth + 1, nodes);
                } else {
                    return false;
                }
            } else {
                values.reverse();
                if values.is_sorted_by(|a, b| a < b) && values.iter().all(|a| a % 2 == 0) {
                    return bfs(deepth + 1, nodes);
                } else {
                    return false;
                }
            }
        }
    }
    bfs(0, &mut nodes)
}

/// 10
pub fn visible_points(points: Vec<Vec<i32>>, angle: i32, location: Vec<i32>) -> i32 {
    let x = location[0];
    let y = location[1];
    let mut angles = vec![];
    let mut ret = 0;
    let mut same_point = 0;

    for point in points {
        let px = point[0] - x;
        let py = point[1] - y;

        if px == 0 && py == 0 {
            same_point += 1;
            continue;
        }

        angles.push(f64::from(py).atan2(f64::from(px)));
    }

    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = angles.len();
    for i in 0..n {
        angles.push(angles[i] + PI * 2.0);
    }

    let mut r = 0;
    for l in 0..n {
        while r < n * 2 && angles[r] - angles[l] <= f64::from(angle) * PI / 180.0 {
            r += 1;
        }
        ret = ret.max((r - l) as i32);
    }

    ret + same_point
}

/// 11
pub fn minimum_one_bit_operations(n: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    let x = (n as f64).log2().floor() as i32;
    (1 << (x + 1)) - 1 - minimum_one_bit_operations(n - (1 << x))
}

/// 14
pub fn max_depth(s: String) -> i32 {
    s.chars()
        .into_iter()
        .fold((0, 0), |mut acc, cur| {
            if cur == '(' {
                acc.1 = acc.1 + 1;
                acc.0 = acc.0.max(acc.1);
            } else if cur == ')' {
                acc.1 = acc.1 - 1;
            }
            acc
        })
        .0
}

/// 15
pub fn maximal_network_rank(n: i32, roads: Vec<Vec<i32>>) -> i32 {
    let n = n as usize;
    let (mut deg, mut con) = (vec![0; n], vec![vec![false; n]; n]);
    roads.iter().for_each(|c| {
        deg[c[0] as usize] += 1;
        deg[c[1] as usize] += 1;
        con[c[0] as usize][c[1] as usize] = true;
        con[c[1] as usize][c[0] as usize] = true;
    });
    let mut max = 0;
    for i in 0..n {
        for j in i + 1..n {
            max = max.max(deg[i] + deg[j] - if con[i][j] == true { 1 } else { 0 })
        }
    }
    max
}

/// 16
pub fn check_palindrome_formation(a: String, b: String) -> bool {
    fn check1(a: &[u8], b: &[u8]) -> bool {
        let (mut i, mut j) = (0, b.len() - 1);
        while i < j && a[i] == b[j] {
            i += 1;
            j -= 1;
        }
        if i >= j {
            return true;
        }
        check2(a, i, j) || check2(b, i, j)
    }

    fn check2(a: &[u8], mut i: usize, mut j: usize) -> bool {
        while i < j && a[i] == a[j] {
            i += 1;
            j -= 1;
        }
        i >= j
    }

    let a = a.as_bytes();
    let b = b.as_bytes();
    check1(a, b) || check1(b, a)
}

/// 19
pub fn trim_mean(mut arr: Vec<i32>) -> f64 {
    let size = arr.len();
    let count = size / 20;
    arr.sort();
    arr[count..size - count]
        .iter()
        .fold(0.0, |acc, &cur| acc + cur as f64)
        / (size as f64 * 0.9)
}

/// 20
pub fn best_coordinate(towers: Vec<Vec<i32>>, radius: i32) -> Vec<i32> {
    let mut ans = (0, Reverse(0), Reverse(0));

    for i in 0..51 {
        for j in 0..51 {
            let mut sum = 0;

            for tower in &towers {
                let (x, y, q) = (tower[0], tower[1], tower[2]);
                let d = (x - i).pow(2) + (y - j).pow(2);

                if d <= radius.pow(2) {
                    sum += ((q as f64) / (1.0 + (d as f64).sqrt())).floor() as i32;
                }
            }

            ans = ans.max((sum, Reverse(i), Reverse(j)))
        }
    }

    vec![ans.1.0, ans.2.0]
}
