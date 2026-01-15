use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    i32::{self},
    rc::Rc,
};

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
pub fn remove_outer_parentheses(s: String) -> String {
    let mut cnt = 0;
    s.chars()
        .filter(|ch| match ch {
            '(' => {
                cnt += 1;
                cnt > 1
            }
            ')' => {
                cnt -= 1;
                cnt > 0
            }
            _ => false,
        })
        .collect::<String>()
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

/// 1023
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

/// 1024
pub fn video_stitching(clips: Vec<Vec<i32>>, time: i32) -> i32 {
    let mut dp = vec![i32::MAX - 1; time as usize + 1];
    dp[0] = 0;
    for t in 1..=time {
        clips.iter().for_each(|c| {
            if c[0] < t && c[1] >= t {
                dp[t as usize] = dp[t as usize].min(dp[c[0] as usize] + 1);
            }
        });
    }
    if dp[time as usize] == i32::MAX - 1 {
        -1
    } else {
        dp[time as usize]
    }
}

/// 1025
pub fn divisor_game(n: i32) -> bool {
    n % 2 == 0
}

/// 1026
pub fn max_ancestor_diff(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut max_diff = i32::MIN;
    fn min_max(
        root: Option<Rc<RefCell<TreeNode>>>,
        max_diff: &mut i32,
    ) -> (Option<i32>, Option<i32>) {
        if let Some(node) = root.as_ref() {
            let root_val = node.borrow().val;
            let (left_min, left_max) = min_max(node.borrow().left.clone(), max_diff);
            let (right_min, right_max) = min_max(node.borrow().right.clone(), max_diff);
            let (mut new_min, mut new_max) = (root_val, root_val);
            if let Some(v) = left_min {
                *max_diff = (*max_diff).max((root_val - v).abs());
                new_min = new_min.min(v)
            }
            if let Some(v) = left_max {
                *max_diff = (*max_diff).max((root_val - v).abs());
                new_max = new_max.max(v)
            }
            if let Some(v) = right_min {
                *max_diff = (*max_diff).max((root_val - v).abs());
                new_min = new_min.min(v)
            }
            if let Some(v) = right_max {
                *max_diff = (*max_diff).max((root_val - v).abs());
                new_max = new_max.max(v)
            }
            (Some(new_min), Some(new_max))
        } else {
            (None, None)
        }
    }
    min_max(root, &mut max_diff);
    max_diff
}

/// 1027
pub fn longest_arith_seq_length(nums: Vec<i32>) -> i32 {
    let mut record: HashMap<(usize, i32), usize> = HashMap::new();
    let mut max = 1;
    for cur in 0..nums.len() {
        for pre in 0..cur {
            let gap = nums[cur] - nums[pre];
            let new_v = match record.get(&(pre, gap)) {
                None => 2,
                Some(v) => v + 1,
            };
            record.insert((cur, gap), new_v);
            max = max.max(new_v);
        }
    }
    max as i32
}

/// 1028
pub fn recover_from_preorder(s: String) -> Option<Rc<RefCell<TreeNode>>> {
    let mut ans: HashMap<i32, Option<Rc<RefCell<TreeNode>>>> = HashMap::new();
    ans.insert(-1, Some(Rc::new(RefCell::new(TreeNode::new(0)))));
    let mut add_tree = |v, p| {
        ans.insert(p, Some(Rc::new(RefCell::new(TreeNode::new(v)))));
        let mut r = ans[&(p - 1)].as_ref().unwrap().borrow_mut();
        if r.left.is_none() {
            r.left = ans[&p].clone();
        } else {
            r.right = ans[&p].clone();
        }
    };
    let mut val = 0;
    let mut dep = 0;
    let mut has_val = false;
    for c in s.chars() {
        if c != '-' {
            val = 10 * val + c as i32 - 48;
            has_val = true;
        } else if has_val {
            add_tree(val, dep);
            val = 0;
            dep = 1;
            has_val = false;
        } else {
            dep += 1;
        }
    }
    add_tree(val, dep);
    ans[&0].clone()
}

//// 1029
pub fn two_city_sched_cost(mut costs: Vec<Vec<i32>>) -> i32 {
    costs.sort_by_key(|c| c[0] - c[1]);
    let length = costs.len();
    costs.iter().enumerate().fold(0, |acc, cur| {
        if cur.0 < length / 2 {
            acc + cur.1[0]
        } else {
            acc + cur.1[1]
        }
    })
}

/// 1030
pub fn all_cells_dist_order(rows: i32, cols: i32, r_center: i32, c_center: i32) -> Vec<Vec<i32>> {
    let mut grid = vec![];
    for i in 0..rows {
        for j in 0..cols {
            grid.push(vec![i, j]);
        }
    }
    grid.sort_by_key(|a| (a[0] - r_center).abs() + (a[1] - c_center).abs());
    grid
}

/// 1031
pub fn max_sum_two_no_overlap(mut nums: Vec<i32>, first_len: i32, second_len: i32) -> i32 {
    (1..nums.len()).for_each(|i| nums[i] += nums[i - 1]);
    let (mut max_first, mut max_second, mut ret) = (
        nums[first_len as usize - 1],
        nums[second_len as usize - 1],
        nums[(first_len + second_len) as usize - 1],
    );
    for i in (first_len + second_len) as usize..nums.len() {
        max_first = max_first
            .max(nums[i - second_len as usize] - nums[i - (first_len + second_len) as usize]);
        max_second = max_second
            .max(nums[i - first_len as usize] - nums[i - (first_len + second_len) as usize]);
        ret = ret.max(
            (max_first + nums[i] - nums[i - second_len as usize])
                .max(max_second + nums[i] - nums[i - first_len as usize]),
        );
    }
    ret
}

/// 1032
#[derive(Debug, Clone)]
struct Trie {
    is_word: bool,
    next: Vec<Option<Box<Trie>>>,
}

impl Trie {
    fn new() -> Self {
        Trie {
            is_word: false,
            next: vec![None; 26],
        }
    }
}

struct StreamChecker {
    max_len: usize,
    curr: String,
    root: Box<Trie>,
}

impl StreamChecker {
    fn new(words: Vec<String>) -> Self {
        let (mut root, mut max_len) = (Box::new(Trie::new()), 0);
        for word in words {
            let mut curr_trie = &mut root;
            max_len = max_len.max(word.len());
            let word_arr = word.as_bytes();
            for i in (0..word_arr.len()).rev() {
                let ch = word_arr[i];
                if curr_trie.next[(ch - b'a') as usize].is_none() {
                    curr_trie.next[(ch - b'a') as usize] = Some(Box::new(Trie::new()));
                }
                curr_trie = curr_trie.next[(ch - b'a') as usize].as_mut().unwrap();
            }
            curr_trie.is_word = true;
        }
        StreamChecker {
            max_len,
            curr: "".to_string(),
            root,
        }
    }

    fn query(&mut self, letter: char) -> bool {
        if self.curr.len() >= self.max_len {
            self.curr.remove(0);
        }
        self.curr.push(letter);
        let mut curr_trie = &mut self.root;
        let curr_arr = self.curr.as_bytes();
        for i in (0..curr_arr.len()).rev() {
            let ch = curr_arr[i];
            if curr_trie.next[(ch - b'a') as usize].is_none() {
                return false;
            }
            curr_trie = curr_trie.next[(ch - b'a') as usize].as_mut().unwrap();
            if curr_trie.is_word {
                return true;
            }
        }
        curr_trie.is_word
    }
}

/// 1033
pub fn num_moves_stones(a: i32, b: i32, c: i32) -> Vec<i32> {
    let mut s = vec![a, b, c];
    s.sort();
    vec![
        if s[2] - s[0] == 2 {
            0
        } else if s[2] - s[1] < 3 || s[1] - s[0] < 3 {
            1
        } else {
            2
        },
        s[2] - s[0] - 2,
    ]
}

/// 1034
pub fn color_border(mut grid: Vec<Vec<i32>>, row: i32, col: i32, color: i32) -> Vec<Vec<i32>> {
    let old_color = grid[row as usize][col as usize];
    let mut searched: HashSet<(usize, usize)> = HashSet::new();
    let mut on_boarder: HashSet<(usize, usize)> = HashSet::new();
    fn dfs(
        grid: &Vec<Vec<i32>>,
        row: usize,
        col: usize,
        old_color: i32,
        searched: &mut HashSet<(usize, usize)>,
        on_boarder: &mut HashSet<(usize, usize)>,
    ) {
        searched.insert((row, col));
        let direction = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        direction
            .iter()
            .map(|d| return ((d.0 + row as i32) as usize, (d.1 + col as i32) as usize))
            .for_each(|(new_r, new_c)| {
                if new_r >= grid.len() || new_c >= grid[0].len() || grid[new_r][new_c] != old_color
                {
                    on_boarder.insert((row, col));
                } else if !searched.contains(&(new_r, new_c)) {
                    dfs(grid, new_r, new_c, old_color, searched, on_boarder);
                }
            });
    }
    dfs(
        &grid,
        row as usize,
        col as usize,
        old_color,
        &mut searched,
        &mut on_boarder,
    );
    on_boarder.iter().for_each(|e| {
        grid[e.0][e.1] = color;
    });
    grid
}

/// 1035
pub fn max_uncrossed_lines(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let mut dp = vec![vec![0; nums2.len() + 1]; nums1.len() + 1];
    for i in 1..=nums1.len() {
        for j in 1..=nums2.len() {
            if nums1[i - 1] == nums2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[nums1.len()][nums2.len()]
}

/// 1036
pub fn is_escape_possible(blocked: Vec<Vec<i32>>, source: Vec<i32>, target: Vec<i32>) -> bool {
    let mut nodes = blocked.iter().collect::<Vec<_>>();
    nodes.push(&source);
    nodes.push(&target);

    let mut dx: Vec<i32> = nodes.iter().map(|x| x[0]).collect();
    let mut dy: Vec<i32> = nodes.iter().map(|x| x[1]).collect();
    dx.sort_unstable();
    dy.sort_unstable();

    let (mut hash_x, mut hash_y) = (HashMap::new(), HashMap::new());

    let discrete = |xy, hash: &mut HashMap<_, _>| -> usize {
        let (mut pos, mut pre) = (0usize, 0);
        for &k in xy {
            match k - pre {
                0 => (),
                1 => pos += 1,
                _ => pos += 2,
            }
            hash.insert(k, pos);
            pre = k;
        }
        pos + 1
    };
    let mut rows = discrete(&dx, &mut hash_x);
    let mut cols = discrete(&dy, &mut hash_y);

    rows += if *dx.last().unwrap() != 999999 { 1 } else { 0 };
    cols += if *dy.last().unwrap() != 999999 { 1 } else { 0 };

    let mut grid = vec![vec![false; cols]; rows];
    blocked.iter().for_each(|x| {
        grid[hash_x[&x[0]]][hash_y[&x[1]]] = true;
    });

    let t = (hash_x[&target[0]], hash_y[&target[1]]);
    let mut q = VecDeque::new();
    q.push_back((hash_x[&source[0]], hash_y[&source[1]]));
    while let Some(pos) = q.pop_front() {
        if pos == t {
            return true;
        }
        for p in [
            (pos.0 - 1, pos.1),
            (pos.0 + 1, pos.1),
            (pos.0, pos.1 - 1),
            (pos.0, pos.1 + 1),
        ] {
            if p.0 < grid.len() && p.1 < grid[0].len() && !grid[p.0][p.1] {
                q.push_back((p.0, p.1));
                grid[p.0][p.1] = true;
            }
        }
    }
    false
}

/// 1037
pub fn is_boomerang(points: Vec<Vec<i32>>) -> bool {
    (points[1][1] - points[0][1]) * (points[2][0] - points[0][0])
        != (points[2][1] - points[0][1]) * (points[1][0] - points[0][0])
}

/// 1038
pub fn bst_to_gst(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    fn reverse(root: Option<Rc<RefCell<TreeNode>>>, total: &mut i32) -> i32 {
        if let Some(node) = root.as_ref() {
            reverse(node.borrow().right.clone(), total);
            *total = *total + node.borrow().val;
            node.borrow_mut().val = *total;
            reverse(node.borrow().left.clone(), total)
        } else {
            0
        }
    }
    let mut total = 0;
    reverse(root.clone(), &mut total);
    root
}

/// 1039
pub fn min_score_triangulation(values: Vec<i32>) -> i32 {
    fn dfs(i: usize, j: usize, v: &Vec<i32>, memo: &mut Vec<Vec<i32>>) -> i32 {
        if i + 1 == j {
            return 0;
        } else if memo[i][j] != -1 {
            return memo[i][j];
        } else {
            let mut res = i32::MAX;
            for k in i + 1..j {
                res = res.min(dfs(i, k, v, memo) + dfs(k, j, v, memo) + v[i] * v[k] * v[j])
            }
            memo[i][j] = res;
            res
        }
    }
    let mut memo = vec![vec![-1; values.len()]; values.len()];
    dfs(0, values.len() - 1, &values, &mut memo)
}

/// 1040
pub fn num_moves_stones_ii(mut stones: Vec<i32>) -> Vec<i32> {
    stones.sort();
    let (mut i, n) = (0, stones.len());
    let (mut l, h) = (
        n,
        (stones[n - 1] - n as i32 + 2 - stones[1]).max(stones[n - 2] - stones[0] - n as i32 + 2),
    );
    for j in 0..n {
        while stones[j] - stones[i] >= n as i32 {
            i += 1;
        }
        if j - i + 1 == n - 1 && stones[j] - stones[i] == n as i32 - 2 {
            l = l.min(2);
        } else {
            l = l.min(n - (j - i + 1));
        }
    }
    vec![l as i32, h]
}

/// 1041
pub fn is_robot_bounded(instructions: String) -> bool {
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum Direction {
        EAST,
        WEST,
        NORTH,
        SOUTH,
    }
    let mut state = ((0, 0), Direction::NORTH);
    instructions.chars().for_each(|c| match c {
        'G' => match state.1 {
            Direction::EAST => state.0 = (state.0.0 + 1, state.0.1),
            Direction::WEST => state.0 = (state.0.0 - 1, state.0.1),
            Direction::NORTH => state.0 = (state.0.0, state.0.1 + 1),
            Direction::SOUTH => state.0 = (state.0.0, state.0.1 - 1),
        },
        'L' => match state.1 {
            Direction::EAST => state.1 = Direction::NORTH,
            Direction::WEST => state.1 = Direction::SOUTH,
            Direction::NORTH => state.1 = Direction::WEST,
            Direction::SOUTH => state.1 = Direction::EAST,
        },
        'R' => match state.1 {
            Direction::EAST => state.1 = Direction::SOUTH,
            Direction::WEST => state.1 = Direction::NORTH,
            Direction::NORTH => state.1 = Direction::EAST,
            Direction::SOUTH => state.1 = Direction::WEST,
        },
        _ => (),
    });
    !(state.0 != (0, 0) && state.1 == Direction::NORTH)
}

/// 1042
pub fn garden_no_adj(n: i32, paths: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::VecDeque;
    let mut path = vec![vec![]; n as usize];
    for p in &paths {
        path[p[0] as usize - 1].push(p[1] as usize - 1);
        path[p[1] as usize - 1].push(p[0] as usize - 1);
    }
    let mut queue = VecDeque::new();
    let mut res = vec![0; n as usize];
    for i in 0..n as usize {
        if res[i] == 0 {
            queue.push_back(i);
            while !queue.is_empty() {
                let len = queue.len();
                for _ in 0..len {
                    let cur = queue.pop_front().unwrap();
                    let mut temp = vec![0; 4];
                    //应该用哪种颜色
                    for &arround in &path[cur] {
                        if res[arround] == 0 {
                            queue.push_back(arround);
                        } else {
                            temp[res[arround] as usize - 1] = 1;
                        }
                    }
                    for i in 0..4 {
                        if temp[i] == 0 {
                            res[cur] = i as i32 + 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    res
}

/// 1043
pub fn max_sum_after_partitioning(nums: Vec<i32>, k: i32) -> i32 {
    let n: usize = nums.len();
    let mut dp: Vec<i32> = vec![0; n];
    for i in 0..n {
        let mut max_value: i32 = 0;
        for cnt in 1..=(std::cmp::min(k, i as i32 + 1)) {
            max_value = std::cmp::max(max_value, nums[i - cnt as usize + 1]);
            dp[i] = std::cmp::max(
                dp[i],
                max_value * cnt
                    + if i < cnt as usize {
                        0
                    } else {
                        dp[i - cnt as usize]
                    },
            );
        }
    }
    dp[n - 1]
}

/// 1044
pub fn longest_dup_substring(s: String) -> String {
    const P: u64 = 131;
    let sv = s.as_bytes();
    let size = s.len();
    let mut p = vec![0; size + 1];
    let mut h = vec![0; size + 1];
    p[0] = 1;
    for i in 1..=size {
        h[i] = h[i - 1] * P + sv[i - 1] as u64;
        p[i] = p[i - 1] * P;
    }
    let mut lo = 0;
    let mut hi = size;
    let mut start_pos = size;
    let mut len = 0;

    fn check(h: &Vec<u64>, p: &Vec<u64>, size: usize, len: usize) -> Option<usize> {
        let mut ss = std::collections::HashSet::new();

        for i in len..=size {
            let code = h[i] - h[i - len] * p[len];
            if ss.contains(&code) {
                return Some(i - len);
            } else {
                ss.insert(code);
            }
        }

        None
    }

    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;

        if let Some(start) = check(&h, &p, size, mid) {
            lo = mid;
            start_pos = start;
            len = mid;
        } else {
            hi = mid - 1;
        }
    }

    if start_pos == size {
        return "".into();
    }

    s[start_pos..start_pos + len].into()
}

/// 1046
pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
    let mut heap = BinaryHeap::new();
    stones.iter().for_each(|&weight| {
        heap.push(weight);
    });
    while heap.len() >= 2 {
        let (first, second) = (heap.pop().unwrap(), heap.pop().unwrap());
        if first != second {
            heap.push(first - second);
        }
    }
    if heap.len() == 0 {
        return 0;
    } else {
        heap.pop().unwrap()
    }
}

/// 1047
pub fn remove_duplicates(s: String) -> String {
    let mut stack = vec![];
    s.chars().for_each(|c| {
        if stack.len() == 0 || *stack.last().unwrap() != c {
            stack.push(c)
        } else {
            stack.pop();
        }
    });
    stack.iter().collect::<String>()
}

/// 1048
pub fn longest_str_chain(mut words: Vec<String>) -> i32 {
    words.sort_by(|a, b| a.len().cmp(&b.len()));
    let mut dp = vec![1; words.len()];
    fn is_precessor(a: &String, b: &String) -> bool {
        if a.len() != b.len() + 1 {
            return false;
        } else {
            for i in 0..a.len() {
                let mut sub_a = a.clone();
                sub_a.remove(i);
                if sub_a == *b {
                    return true;
                }
            }
        }
        false
    }
    for i in 1..words.len() {
        for j in 0..i {
            if is_precessor(&words[i], &words[j]) {
                dp[i] = dp[i].max(dp[j] + 1)
            }
        }
    }
    *dp.iter().max().unwrap()
}

/// 1049
pub fn last_stone_weight_ii(stones: Vec<i32>) -> i32 {
    let sum: i32 = stones.iter().sum();
    let target = sum as usize / 2;
    let mut dp = vec![0; target + 1];
    stones.iter().for_each(|&n| {
        (n as usize..=target).rev().for_each(|i| {
            dp[i] = dp[i].max(dp[i - n as usize] + n);
        });
    });
    sum - 2 * dp[target]
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
