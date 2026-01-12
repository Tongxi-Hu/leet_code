use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use crate::common::{ListNode, TreeNode};

/// 1001
pub fn grid_illumination(n: i32, lamps: Vec<Vec<i32>>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    const DIRS: [[i32; 2]; 9] = [
        [0, 0],
        [1, 1],
        [-1, -1],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, 1],
        [1, -1],
    ];

    use std::collections::{HashMap, HashSet};
    fn match_rule(cache: &HashMap<i32, i32>, grid: &i32) -> bool {
        if let Some(v) = cache.get(grid) {
            *v > 0
        } else {
            false
        }
    }

    let (mut axis_x, mut axis_y, mut axis_x_to_y, mut axis_y_to_x, mut is_bright, mut ret) = (
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashSet::new(),
        Vec::new(),
    );

    for lamp in lamps {
        let (x, y) = (lamp[0], lamp[1]);
        if is_bright.contains(&(x, y)) {
            continue;
        }

        is_bright.insert((x, y));
        *axis_x.entry(x).or_insert(0) += 1;
        *axis_y.entry(y).or_insert(0) += 1;
        *axis_x_to_y.entry(x - y).or_insert(0) += 1;
        *axis_y_to_x.entry(x + y).or_insert(0) += 1;
    }

    for query in queries {
        let (x, y) = (query[0], query[1]);
        ret.push(
            (match_rule(&axis_x, &x)
                || match_rule(&axis_y, &y)
                || match_rule(&axis_x_to_y, &(x - y))
                || match_rule(&axis_y_to_x, &(x + y))) as i32,
        );
        for dir in DIRS {
            let (tx, ty) = (x + dir[0], y + dir[1]);
            if tx < 0 || tx >= n || ty < 0 || ty >= n || !is_bright.contains(&(tx, ty)) {
                continue;
            }
            is_bright.remove(&(tx, ty));
            *axis_x.get_mut(&tx).unwrap() -= 1;
            *axis_y.get_mut(&ty).unwrap() -= 1;
            *axis_x_to_y.get_mut(&(tx - ty)).unwrap() -= 1;
            *axis_y_to_x.get_mut(&(tx + ty)).unwrap() -= 1;
        }
    }
    ret
}

/// 1002
pub fn common_chars(words: Vec<String>) -> Vec<String> {
    let mut min_count = vec![usize::MAX; 26];
    words.into_iter().for_each(|word| {
        let mut count = vec![0; 26];
        word.chars().for_each(|c| {
            let index = c as usize - 'a' as usize;
            count[index] = count[index] + 1;
        });
        min_count = min_count
            .iter()
            .zip(count)
            .map(|(min, c)| *min.min(&c))
            .collect::<Vec<usize>>();
    });
    let mut common = vec![];
    min_count.iter().enumerate().for_each(|(i, c)| {
        if *c != 0 {
            for _ in 0..*c {
                common.push((('a' as u8 + i as u8) as char).to_string());
            }
        }
    });
    common
}

/// 1003
pub fn is_valid(s: String) -> bool {
    let mut stack: Vec<char> = vec![];
    s.chars().for_each(|c| {
        if c == 'c'
            && stack.len() >= 2
            && stack[stack.len() - 1] == 'b'
            && stack[stack.len() - 2] == 'a'
        {
            stack.pop();
            stack.pop();
        } else {
            stack.push(c)
        }
    });
    stack.len() == 0
}

/// 1004
pub fn longest_ones(nums: Vec<i32>, k: i32) -> i32 {
    let mut zero_count = 0;
    let mut max_len = 0;

    let mut l = 0;
    for (i, &num) in nums.iter().enumerate() {
        if num == 0 {
            zero_count += 1;
        }

        while zero_count > k {
            if nums[l] == 0 {
                zero_count -= 1;
            }
            l += 1;
        }

        max_len = i32::max(max_len, (i - l + 1) as i32);
    }

    max_len
}

/// 1005
pub fn largest_sum_after_k_negations(nums: Vec<i32>, k: i32) -> i32 {
    let mut a = nums;
    a.sort_unstable();
    let mut k = k;
    for i in a.iter_mut() {
        if k > 0 && *i < 0 {
            *i *= -1;
            k -= 1;
        } else {
            break;
        }
    }
    a.sort_unstable();
    a[0] *= if k & 1 == 1 { -1 } else { 1 };
    a.into_iter().sum()
}

/// 1006
pub fn clumsy(n: i32) -> i32 {
    let mut flag = 0;
    let mut res = 0;
    let mut tmp = n;
    let mut cur = n;
    cur -= 1;

    while cur > 0 {
        match flag {
            0 => tmp *= cur,
            1 => tmp /= cur,
            2 => {
                res = res + tmp + cur;
                tmp = 0;
            }
            3 => {
                tmp = -cur;
            }
            _ => {}
        }
        flag += 1;
        if flag > 3 {
            flag = 0;
        }
        cur -= 1;
    }
    if tmp != 0 {
        res += tmp;
    }
    res
}

/// 1007
pub fn min_domino_rotations(tops: Vec<i32>, bottoms: Vec<i32>) -> i32 {
    fn check(x: i32, tops: &[i32], bottoms: &[i32], n: usize) -> i32 {
        let (mut rotations_a, mut rotations_b) = (0, 0);
        for i in 0..n {
            if tops[i] != x && bottoms[i] != x {
                return -1;
            } else if tops[i] != x {
                rotations_a += 1;
            } else if bottoms[i] != x {
                rotations_b += 1;
            }
        }
        rotations_a.min(rotations_b)
    }
    let n = tops.len();
    let rotations = check(tops[0], &tops, &bottoms, n);
    if rotations != -1 || tops[0] == bottoms[0] {
        return rotations;
    }
    check(bottoms[0], &tops, &bottoms, n)
}

/// 1008
pub fn bst_from_preorder(preorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn construct_bst_in_range(
        data: &Vec<i32>,
        left: usize,
        right: usize,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        println!("{},{}", left, right);
        if left > right {
            None
        } else {
            let mut root = Some(Rc::new(RefCell::new(TreeNode::new(data[left]))));
            let mut left_new = left;
            while left_new <= right {
                if data[left_new] > data[left] {
                    break;
                } else {
                    left_new = left_new + 1;
                }
            }
            root.as_mut().unwrap().borrow_mut().left =
                construct_bst_in_range(data, left + 1, left_new - 1);
            root.as_mut().unwrap().borrow_mut().right =
                construct_bst_in_range(data, left_new, right);
            root
        }
    }
    construct_bst_in_range(&preorder, 0, preorder.len() - 1)
}

/// 1009
pub fn bitwise_complement(n: i32) -> i32 {
    if n == 0 {
        return 1;
    }

    let mut t = n;
    let mut m = 1;

    while t != 0 {
        t >>= 1;
        m <<= 1;
    }

    n ^ m - 1
}

/// 1010
pub fn num_pairs_divisible_by60(time: Vec<i32>) -> i64 {
    let mut remainder = vec![0; 60];
    time.iter().for_each(|&t| {
        let r = (t as usize) % 60;
        remainder[r] = remainder[r] + 1;
    });
    let mut total = 0;
    for i in 1..30 {
        total = total + remainder[i] * remainder[60 - i]
    }
    total = total + remainder[0] * (remainder[0] - 1) / 2 + remainder[30] * (remainder[30] - 1) / 2;
    total
}

/// 1011
pub fn ship_within_days(weights: Vec<i32>, days: i32) -> i32 {
    let min_capacity = *weights.iter().max().unwrap();
    let max_capacity = weights.iter().sum();

    let mut left = min_capacity;
    let mut right = max_capacity;
    fn get_days(weights: &Vec<i32>, max_capacity: i32) -> i32 {
        let mut days = 0;
        let mut curr_capacity = 0;
        for weight in weights.iter() {
            if curr_capacity <= max_capacity && curr_capacity + weight > max_capacity {
                days += 1;
                curr_capacity = *weight;
            } else {
                curr_capacity += weight;
            }
        }
        days + if curr_capacity != 0 { 1 } else { 0 }
    }
    while left <= right {
        let mid = left + (right - left) / 2;
        match get_days(&weights, mid).cmp(&days) {
            std::cmp::Ordering::Greater => left = mid + 1,
            _ => right = mid - 1,
        }
    }
    left
}

/// 1012
pub fn num_dup_digits_at_most_n(n: i32) -> i32 {
    let nums: Vec<i32> = n.to_string().chars().map(|c| c as i32 - 48).collect();
    let pre_sum = [0, 9, 90, 738, 5274, 32490, 168570, 712890, 2345850, 5611770];
    let nums_len: usize = nums.len();
    let mut res: i32 = pre_sum[nums_len - 1];
    let mut index: usize = 0;
    let mut used: Vec<bool> = vec![false; 10];
    let get_fact = |mut f: i32, cnt: i32| -> i32 {
        let mut fact: i32 = 1;
        for _ in 0..cnt {
            fact *= f;
            f -= 1;
        }
        fact
    };
    loop {
        if index == nums_len {
            res += 1;
            break;
        }
        for this_num in if index == 0 { 1 } else { 0 }..nums[index] {
            if used[this_num as usize] {
                continue;
            }
            res += get_fact(10 - index as i32 - 1, nums_len as i32 - index as i32 - 1);
        }
        if used[nums[index] as usize] {
            break;
        }
        used[nums[index] as usize] = true;
        index += 1;
    }
    n - res
}

/// 1013
pub fn can_three_parts_equal_sum(arr: Vec<i32>) -> bool {
    let total = arr.iter().sum::<i32>();
    let mut parts = 0;
    let mut cnt = 0;

    if total % 3 != 0 {
        return false;
    }

    for i in 0..arr.len() {
        cnt += arr[i];

        if cnt == total / 3 {
            parts += 1;

            if parts == 2 {
                return i < arr.len() - 1;
            }

            cnt = 0;
        }
    }

    false
}

/// 1014
pub fn max_score_sightseeing_pair(values: Vec<i32>) -> i32 {
    let mut ans = 0;
    let mut mx = values[0];
    for j in 1..values.len() {
        ans = ans.max(mx + values[j] - j as i32);
        mx = mx.max(values[j] + j as i32);
    }
    ans
}

/// 1015
pub fn smallest_repunit_div_by_k(k: i32) -> i32 {
    if k % 2 == 0 || k % 5 == 0 {
        return -1;
    }
    let mut offset = 1 % k;
    for i in 1..=k {
        if offset == 0 {
            return i;
        }
        offset = (offset * 10 + 1) % k;
    }
    -1
}

/// 1016
pub fn query_string(s: String, n: i32) -> bool {
    (1..=n).all(|c| {
        let binary_string: String = format!("{:b}", c);
        s.contains(&binary_string)
    })
}

/// 1017
pub fn base_neg2(n: i32) -> String {
    if n == 0 || n == 1 {
        n.to_string()
    } else {
        format!("{}{}", base_neg2(-(n >> 1)), n & 1)
    }
}

/// 1018
pub fn prefixes_div_by5(a: Vec<i32>) -> Vec<bool> {
    let mut ret = Vec::new();
    let mut d: u128 = 0;
    for i in a {
        d = d * 2 + i as u128;
        if d % 5 == 0 {
            d = 0;
            ret.push(true)
        } else {
            ret.push(false)
        }
    }
    ret
}

/// 1019
pub fn next_larger_nodes(head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut vals = vec![];
    let mut cur = &head;
    while let Some(node) = cur.as_ref() {
        vals.push(node.val);
        cur = &node.next;
    }
    let mut mono_stack: VecDeque<(usize, i32)> = VecDeque::new();
    let mut ans = vec![0; vals.len()];
    vals.iter().enumerate().for_each(|(i, &v)| {
        while mono_stack.back().unwrap_or(&(0, i32::MAX)).1 < v {
            let node = mono_stack.pop_back().unwrap();
            ans[node.0] = v;
        }
        mono_stack.push_back((i, v));
    });
    if mono_stack.len() > 0 {
        for i in mono_stack {
            ans[i.0] = 0;
        }
    }
    ans
}

/// 1020
pub fn num_enclaves(mut grid: Vec<Vec<i32>>) -> i32 {
    let (height, width) = (grid.len(), grid[0].len());
    fn attach_to_boundary(
        grid: &mut Vec<Vec<i32>>,
        width: usize,
        height: usize,
        position: (usize, usize),
    ) {
        if grid[position.0][position.1] == 1 {
            grid[position.0][position.1] = -1;
            let direction = vec![(-1, 0), (1, 0), (0, -1), (0, 1)];
            for d in direction.iter() {
                let new_position = (
                    (position.0 as i32 + d.0) as usize,
                    (position.1 as i32 + d.1) as usize,
                );
                if new_position.0 < height && new_position.1 < width {
                    attach_to_boundary(grid, width, height, new_position);
                }
            }
        }
    }
    for i in 0..height {
        attach_to_boundary(&mut grid, width, height, (i, 0));
        attach_to_boundary(&mut grid, width, height, (i, width - 1));
    }
    for j in 0..width {
        attach_to_boundary(&mut grid, width, height, (0, j));
        attach_to_boundary(&mut grid, width, height, (height - 1, j));
    }
    grid.iter().fold(0, |acc, cur| {
        cur.iter().fold(0, |a, &c| if c == 1 { a + c } else { a }) + acc
    })
}

/// 1021
pub fn camel_match(queries: Vec<String>, pattern: String) -> Vec<bool> {
    fn is_match(query: &str, pattern: &str) -> bool {
        let (mut i, query_arr, pattern_arr) = (0, query.as_bytes(), pattern.as_bytes());
        for &ch in query_arr {
            if i < pattern_arr.len() && ch == pattern_arr[i] {
                i += 1;
            } else if ch < b'a' {
                return false;
            }
        }
        i == pattern.len()
    }
    queries
        .into_iter()
        .map(|query| is_match(query.as_str(), pattern.as_str()))
        .collect()
}

/// 1022
pub fn sum_root_to_leaf(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn root_to_leaf_binary(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<String> {
        if let Some(node) = root.as_ref() {
            let mut left = root_to_leaf_binary(node.borrow().left.clone());
            let mut right = root_to_leaf_binary(node.borrow().right.clone());
            left.append(&mut right);
            if left.len() == 0 {
                return vec![node.borrow().val.to_string()];
            }
            return left
                .iter()
                .map(|s| node.borrow().val.to_string() + s)
                .collect();
        } else {
            vec![]
        }
    }
    root_to_leaf_binary(root)
        .iter()
        .fold(0, |acc, cur| acc + i32::from_str_radix(cur, 2).unwrap())
}

#[test]
fn test_1100() {
    println!(
        "{:?}",
        num_enclaves(vec![
            vec![0, 0, 0, 0],
            vec![1, 0, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ])
    );
}
