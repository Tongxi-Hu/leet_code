use core::f64;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    rc::Rc,
};

use crate::common::TreeNode;

/// p605
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    let m = flowerbed.len();
    let mut n = n;
    let mut i = 0;
    while i < m {
        if (i == 0 || flowerbed[i - 1] == 0)
            && flowerbed[i] == 0
            && (i == m - 1 || flowerbed[i + 1] == 0)
        {
            n -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    n <= 0
}

/// p606
pub fn tree2str(root: Option<Rc<RefCell<TreeNode>>>) -> String {
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> Option<String> {
        if let Some(node) = root.as_ref() {
            let mut res = String::new();
            let v = node.borrow().val;
            res = res + &v.to_string();
            let left_res = dfs(&node.borrow().left);
            let right_res = dfs(&node.borrow().right);
            if let Some(left_val) = left_res.as_ref() {
                res = res + "(";
                res = res + &left_val.to_string();
                res = res + ")";
            }
            if let Some(right_val) = right_res {
                if left_res == None {
                    res = res + "()";
                }
                res = res + "(";
                res = res + &right_val.to_string();
                res = res + ")";
            }
            return Some(res);
        } else {
            None
        }
    }
    dfs(&root).unwrap_or("".to_string())
}

/// p609
pub fn find_duplicate(paths: Vec<String>) -> Vec<Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for path in paths.into_iter() {
        let vals = path.split(' ').collect::<Vec<_>>();
        for &s in vals.iter().skip(1) {
            let idx = s.find('(').unwrap();
            let cur_path = vals[0].to_string() + "/" + &s[..idx];
            let cur_content = s[idx..].to_string();
            map.entry(cur_content).or_insert(Vec::new()).push(cur_path);
        }
    }

    map.into_values()
        .filter(|vals| vals.len() > 1)
        .collect::<Vec<_>>()
}

/// p611
pub fn triangle_number(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort();
    let length = nums.len();
    if length < 3 {
        return 0;
    }
    let mut count = 0;
    for i in 0..length - 2 {
        for j in i + 1..length - 1 {
            for k in j + 1..length {
                if nums[i] + nums[j] > nums[k] {
                    count = count + 1;
                }
            }
        }
    }
    count
}

/// p617
pub fn merge_trees(
    root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(
        root1: &Option<Rc<RefCell<TreeNode>>>,
        root2: &Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        match (root1.as_ref(), root2.as_ref()) {
            (None, None) => None,
            (Some(node1), Some(node2)) => {
                let mut node = TreeNode::new(node1.borrow().val + node2.borrow().val);
                node.left = dfs(&node1.borrow().left, &node2.borrow().left);
                node.right = dfs(&node1.borrow().right, &node2.borrow().right);
                Some(Rc::new(RefCell::new(node)))
            }
            (Some(node1), None) => Some(node1.clone()),
            (None, Some(node2)) => Some(node2.clone()),
        }
    }
    dfs(&root1, &root2)
}

/// p621
pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
    let mut counts = [0; 26];
    let mut max_count = 0;
    let mut count = 0;

    for idx in tasks.iter().map(|&task| ((task as u8) - b'A') as usize) {
        counts[idx] += 1;
        if counts[idx] > max_count {
            count = 1;
            max_count = counts[idx];
        } else if counts[idx] == max_count {
            count += 1;
        }
    }

    i32::max(tasks.len() as i32, count + (max_count - 1) * (n + 1))
}

/// p622
struct MyCircularQueue {
    arr: Vec<i32>,
    rear: usize,
    curr_size: usize,
}

impl MyCircularQueue {
    fn new(k: i32) -> Self {
        MyCircularQueue {
            arr: vec![0; k as usize],
            rear: 0,
            curr_size: 0,
        }
    }

    fn en_queue(&mut self, value: i32) -> bool {
        if self.is_full() {
            false
        } else {
            self.arr[self.rear] = value;
            self.curr_size += 1;
            self.rear = (self.rear + 1) % self.arr.len();
            true
        }
    }

    fn de_queue(&mut self) -> bool {
        if self.is_empty() {
            false
        } else {
            self.curr_size -= 1;
            true
        }
    }

    fn front(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - self.curr_size + self.arr.len()) % self.arr.len()]
        }
    }

    fn rear(&self) -> i32 {
        if self.is_empty() {
            -1
        } else {
            self.arr[(self.rear - 1 + self.arr.len()) % self.arr.len()]
        }
    }

    fn is_empty(&self) -> bool {
        self.curr_size == 0
    }

    fn is_full(&self) -> bool {
        self.curr_size == self.arr.len()
    }
}

///p623
pub fn add_one_row(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
    depth: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    use std::collections::VecDeque;
    if depth == 1 {
        let mut node = TreeNode::new(val);
        node.left = root;
        return Some(Rc::new(RefCell::new(node)));
    }
    let mut queue = VecDeque::new();
    queue.push_back(root.clone());
    for _ in 0..depth - 2 {
        let size = queue.len();
        for _ in 0..size {
            if let Some(node) = queue.pop_front() {
                if node.as_ref().unwrap().borrow().left.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().left.clone());
                }
                if node.as_ref().unwrap().borrow().right.is_some() {
                    queue.push_back(node.as_ref().unwrap().borrow().right.clone());
                }
            }
        }
    }
    while !queue.is_empty() {
        if let Some(node) = queue.pop_front() {
            let mut new_node = TreeNode::new(val);
            new_node.left = node.as_ref().unwrap().borrow().left.clone();
            node.as_ref().unwrap().borrow_mut().left = Some(Rc::new(RefCell::new(new_node)));
            let mut new_node = TreeNode::new(val);
            new_node.right = node.as_ref().unwrap().borrow().right.clone();
            node.as_ref().unwrap().borrow_mut().right = Some(Rc::new(RefCell::new(new_node)));
        }
    }
    root
}

/// p624
pub fn max_distance(arrays: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let mut min_val = arrays[0][0];
    let mut max_val = arrays[0][arrays[0].len() - 1];
    for i in 1..arrays.len() {
        let n = arrays[i].len();
        res = res.max((arrays[i][n - 1] - min_val).abs());
        res = res.max((max_val - arrays[i][0]).abs());
        min_val = min_val.min(arrays[i][0]);
        max_val = max_val.max(arrays[i][n - 1]);
    }
    res
}

/// p628
pub fn maximum_product(mut nums: Vec<i32>) -> i32 {
    let n = nums.len();
    nums.sort_unstable();

    let mut max_num = nums[n - 3] * nums[n - 2] * nums[n - 1];
    if nums[1] < 0 {
        max_num = i32::max(max_num, nums[0] * nums[1] * nums[n - 1]);
    }

    max_num
}

/// p629
const MOD: i64 = 1_0000_0000_7;

pub fn k_inverse_pairs(n: i32, k: i32) -> i32 {
    let (n, k) = (n as usize, k as usize);

    let mut dp = vec![vec![0_i64; k + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = 1;
    }

    for i in 1..=n {
        for j in 1..=k {
            dp[i][j] = if j >= i {
                (dp[i][j - 1] + MOD - dp[i - 1][j - i] + dp[i - 1][j]) % MOD
            } else {
                (dp[i][j - 1] + dp[i - 1][j]) % MOD
            };
        }
    }

    dp[n][k] as i32
}

/// p630
pub fn schedule_course(courses: Vec<Vec<i32>>) -> i32 {
    let mut courses: Vec<_> = courses
        .into_iter()
        .filter(|course| course[0] <= course[1])
        .collect();
    courses.sort_by_key(|c| c[1]);

    if courses.is_empty() {
        return 0;
    }

    let mut cur_time = 0;
    let mut pq = BinaryHeap::new();

    for course in courses {
        let (duration, ddl) = (course[0], course[1]);
        if cur_time + duration <= ddl {
            cur_time += duration;
            pq.push(duration);
        } else {
            if let Some(&d) = pq.peek() {
                if duration < d {
                    cur_time = cur_time - d + duration;
                    pq.pop();
                    pq.push(duration);
                }
            }
        }
    }

    pq.len() as i32
}

/// p632
pub fn smallest_range(nums: Vec<Vec<i32>>) -> Vec<i32> {
    let mut range_left = 0;
    let mut range_right = i32::MAX;
    let size = nums.len();
    let mut next = vec![0; size];
    let mut max_value = i32::MIN;
    let mut pq = BinaryHeap::new();

    for i in 0..size {
        max_value = max_value.max(nums[i][0]);
        pq.push(std::cmp::Reverse((nums[i][0], i)));
    }

    while let Some(std::cmp::Reverse((min_value, row))) = pq.pop() {
        if max_value - min_value < range_right - range_left {
            range_left = min_value;
            range_right = max_value;
        }
        if next[row] == nums[row].len() - 1 {
            break;
        }
        next[row] += 1;
        max_value = max_value.max(nums[row][next[row]]);
        pq.push(std::cmp::Reverse((nums[row][next[row]], row)));
    }

    vec![range_left, range_right]
}

/// p633
pub fn judge_square_sum(c: i32) -> bool {
    let (mut a, mut b) = (0, (c as f64).sqrt() as i32);
    while a <= b {
        if a * a == c - b * b {
            return true;
        }
        if a * a < c - b * b {
            a += 1
        } else {
            b -= 1;
        }
    }
    false
}

/// p636
pub fn exclusive_time(n: i32, logs: Vec<String>) -> Vec<i32> {
    let (mut ret, mut queue) = (vec![0; n as usize], Vec::new());
    for log in logs {
        let arr = log.split(":").collect::<Vec<_>>();
        let (num, t) = (
            arr[0].parse::<i32>().unwrap(),
            arr[2].parse::<i32>().unwrap(),
        );
        if "start" == arr[1] {
            queue.push((num, t))
        } else {
            if let Some(last) = queue.pop() {
                let cost = t - last.1 + 1;
                ret[num as usize] += cost;
                let _ = queue.last().map_or((), |last| ret[last.0 as usize] -= cost);
            }
        }
    }
    ret
}

/// p637
pub fn average_of_levels(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<f64> {
    if let Some(node) = root.as_ref() {
        let mut current_level: Vec<Rc<RefCell<TreeNode>>> = vec![node.clone()];
        let mut average: Vec<f64> = vec![];
        loop {
            let length: usize = current_level.len();
            let mut current_sum: f64 = 0.0;
            for _ in 0..length {
                let node = current_level.remove(0).clone();
                current_sum = (node.borrow().val as f64) + current_sum;
                if let Some(left) = node.borrow().left.as_ref() {
                    current_level.push(left.clone())
                }
                if let Some(right) = node.borrow().right.as_ref() {
                    current_level.push(right.clone());
                }
            }
            average.push(current_sum / (length as f64));
            if current_level.len() == 0 {
                break;
            }
        }
        return average;
    } else {
        vec![]
    }
}

/// p638
pub fn shopping_offers(price: Vec<i32>, special: Vec<Vec<i32>>, needs: Vec<i32>) -> i32 {
    let n: usize = price.len();
    fn dfs(price: &Vec<i32>, special: &Vec<Vec<i32>>, needs: &Vec<i32>, i: usize, n: usize) -> i32 {
        let mut ans = 0;
        for j in 0..n {
            ans += needs[j] * price[j];
        }
        if special.is_empty() {
            return ans;
        }
        let (curr, mut next_needs) = (&special[i], vec![]);
        for j in 0..n {
            if curr[j] > needs[j] {
                break;
            }
            next_needs.push(needs[j] - curr[j]);
        }

        if next_needs.len() == n {
            ans = ans.min(curr[n] + dfs(price, special, &next_needs, i, n));
        }
        if i + 1 < special.len() {
            ans = ans.min(dfs(price, special, needs, i + 1, n));
        }
        ans
    }

    let mut filter_specials = Vec::new();
    for s in special {
        let (mut total_cnt, mut total_price) = (0, 0);
        for i in 0..n {
            total_cnt += s[i];
            total_price += price[i] * s[i];
        }
        if total_cnt > 0 && total_price > s[s.len() - 1] {
            filter_specials.push(s);
        }
    }

    dfs(&price, &filter_specials, &needs, 0, n)
}

/// p639
pub fn num_decodings(s: String) -> i32 {
    let m = 1_000_000_007;
    let (mut a, mut b, mut c) = (0i64, 1i64, 0i64);
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        c = b * check1digit(bytes[i]) % m;
        if i > 0 {
            c += a * check2digits(bytes[i - 1], bytes[i]) % m;
        }
        c %= m;
        a = b;
        b = c;
    }
    c as i32
}

fn check1digit(b: u8) -> i64 {
    match b {
        b'*' => 9,
        b'0' => 0,
        _ => 1,
    }
}

fn check2digits(b1: u8, b2: u8) -> i64 {
    match (b1, b2) {
        (b'*', b'*') => 15,
        (b'*', x) => {
            if x > b'6' {
                1
            } else {
                2
            }
        }
        (x, b'*') => match x {
            b'1' => 9,
            b'2' => 6,
            _ => 0,
        },
        (b'1', _) => 1,
        (b'2', b'0'..=b'6') => 1,
        _ => 0,
    }
}

/// p640
pub fn solve_equation(equation: String) -> String {
    let arr = equation.split("=").collect::<Vec<_>>();

    fn parse_expression(expression: &str) -> (i32, i32) {
        let expression = expression.replace("+x", "+1x").replace("-x", "-1x");
        let expression = if expression.chars().nth(0) == Some('x') {
            format!("+1{}", expression)
        } else {
            expression
        };
        let expression = expression.replace("-", "+-");
        let split_arr = expression
            .split("+")
            .filter(|x| x.len() > 0)
            .collect::<Vec<_>>();

        let prefix = split_arr
            .iter()
            .filter(|x| x.chars().rev().nth(0) == Some('x'))
            .map(|x| x.trim_end_matches("x"))
            .map(|x| x.parse::<i32>().unwrap())
            .sum::<i32>();
        let value = split_arr
            .iter()
            .filter(|x| x.chars().rev().nth(0) != Some('x'))
            .map(|x| x.parse::<i32>().unwrap())
            .sum::<i32>();
        return (prefix, value);
    }

    let (mut left, mut right) = (parse_expression(arr[0]), parse_expression(arr[1]));
    left.0 -= right.0;
    right.1 -= left.1;
    match (left.0, right.1) {
        (0, 0) => "Infinite solutions".to_string(),
        (0, _) => "No solution".to_string(),
        (x, y) => format!("x={}", y / x),
    }
}

/// p643
pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
    let k = k as usize;
    let mut sum: i32 = nums[0..k].iter().sum();
    let mut max = sum;
    for i in k..nums.len() {
        sum = sum - nums[i - k] + nums[i];
        max = max.max(sum);
    }
    max as f64 / k as f64
}

/// p645
pub fn find_error_nums(nums: Vec<i32>) -> Vec<i32> {
    let mut cnt = vec![0; nums.len() + 1];
    for n in nums {
        cnt[n as usize] += 1;
    }
    let mut ans = vec![0; 2];
    for (i, &c) in cnt.iter().enumerate().skip(1) {
        if c == 2 {
            ans[0] = i as i32;
        } else if c == 0 {
            ans[1] = i as i32;
        }
    }
    ans
}

/// p646
pub fn find_longest_chain(pairs: Vec<Vec<i32>>) -> i32 {
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a[1].cmp(&b[1]));
    let mut cur = i32::MIN;
    let mut count = 0;
    pairs.iter().for_each(|p| {
        if cur < p[0] {
            cur = p[1];
            count = count + 1;
        }
    });
    count
}

/// p647
pub fn count_substrings(s: String) -> i32 {
    let mut ans = 0;
    let s = s.chars().collect::<Vec<_>>();
    let mut dp = vec![vec![false; s.len()]; s.len()];
    for i in (0..s.len()).rev() {
        dp[i][i] = true;
        ans += 1;
        for j in i + 1..s.len() {
            dp[i][j] = s[i] == s[j] && if i + 1 == j { true } else { dp[i + 1][j - 1] };
            ans += if dp[i][j] { 1 } else { 0 };
        }
    }
    ans
}

/// p648
pub fn replace_words(dictionary: Vec<String>, sentence: String) -> String {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let mut dictionary = dictionary;
    dictionary.sort_by(|a, b| a.len().cmp(&b.len()));
    let mut ans: Vec<&str> = vec![];
    for word in words {
        let mut modified = false;
        'b: for root in &dictionary {
            if word.starts_with(root) {
                ans.push(root);
                modified = true;
                break 'b;
            }
        }
        if modified == false {
            ans.push(word)
        }
    }
    ans.join(&" ")
}

/// p649
pub fn predict_party_victory(mut senate: String) -> String {
    let (mut r_alive, mut d_alive) = (true, true);
    let mut mark: i32 = 0;
    while r_alive && d_alive {
        (r_alive, d_alive) = (false, false);
        for c in unsafe { senate.as_bytes_mut() } {
            match c {
                b'0' => {}
                b'R' => {
                    r_alive = true;
                    if mark < 0 {
                        *c = b'0';
                    }
                    mark += 1;
                }
                b'D' => {
                    d_alive = true;
                    if mark > 0 {
                        *c = b'0';
                    }
                    mark -= 1;
                }
                _ => unsafe { std::hint::unreachable_unchecked() },
            }
        }
    }
    match mark.cmp(&0) {
        std::cmp::Ordering::Greater => "Radiant".to_owned(),
        std::cmp::Ordering::Less => "Dire".to_owned(),
        std::cmp::Ordering::Equal => unreachable!(),
    }
}

/// p650
pub fn min_steps(n: i32) -> i32 {
    let n = n as usize;
    let mut dp: Vec<usize> = vec![usize::MAX; n + 1];
    dp[1] = 0;
    for i in 2..=n {
        for j in 1..i {
            if i % j == 0 {
                dp[i] = dp[i].min(dp[j] + i / j)
            }
        }
    }
    dp[n] as i32
}

/// p652
pub fn find_duplicate_subtrees(
    root: Option<Rc<RefCell<TreeNode>>>,
) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
    let mut ans = vec![];
    let mut cnt = HashMap::new();

    fn dfs(
        ans: &mut Vec<Option<Rc<RefCell<TreeNode>>>>,
        cnt: &mut HashMap<String, i32>,
        root: Option<Rc<RefCell<TreeNode>>>,
    ) -> String {
        if let Some(r) = root {
            let left = dfs(ans, cnt, r.borrow().left.clone());
            let right = dfs(ans, cnt, r.borrow().right.clone());
            let key = format!("{} {} {}", r.borrow().val, left, right);

            *cnt.entry(key.clone()).or_insert(0) += 1;

            if let Some(&v) = cnt.get(&key) {
                if v == 2 {
                    ans.push(Some(r));
                }
            }

            key
        } else {
            ",".to_string()
        }
    }

    dfs(&mut ans, &mut cnt, root);

    ans
}

/// p653
pub fn find_target(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> bool {
    let mut vals: Vec<i32> = vec![];
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, vals: &mut Vec<i32>) {
        if let Some(node) = root.as_ref() {
            dfs(&node.borrow().left, vals);
            vals.push(node.borrow().val);
            dfs(&node.borrow().right, vals);
        }
    }
    dfs(&root, &mut vals);
    let mut left = 0;
    let mut right = vals.len() - 1;
    while left < right {
        let sum = vals[left] + vals[right];
        if sum == k {
            return true;
        } else if sum < k {
            left = left + 1;
        } else {
            right = right - 1;
        }
    }
    false
}

/// p654
pub fn construct_maximum_binary_tree(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn build_tree(nums: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        let length = nums.len();
        if length == 0 {
            return None;
        } else if let Some((i, max)) = nums.iter().enumerate().max_by_key(|(_, v)| *v) {
            let node = Rc::new(RefCell::new(TreeNode::new(*max)));
            if i > 0 {
                node.borrow_mut().left = build_tree(&nums[0..i]);
            }
            if i + 1 < length {
                node.borrow_mut().right = build_tree(&nums[i + 1..]);
            }
            return Some(node);
        }
        None
    }
    build_tree(&nums)
}

/// p655
pub fn print_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<String>> {
    fn get_depth(root: &Option<Rc<RefCell<TreeNode>>>) -> usize {
        if let Some(r) = root {
            1 + get_depth(&r.borrow().left).max(get_depth(&r.borrow().right))
        } else {
            0
        }
    }

    let n = get_depth(&root);
    let m = (1 << n) - 1;
    let mut queue = vec![];
    let mut grid = vec![vec!["".to_string(); m]; n];
    if let Some(r) = root {
        queue.push((r, 0, (m - 1) / 2));

        while queue.len() > 0 {
            let mut tmp = vec![];

            for i in 0..queue.len() {
                let (ref node, row, col) = queue[i];
                grid[row][col] = format!("{}", node.borrow().val);

                if let Some(left) = node.borrow_mut().left.take() {
                    tmp.push((left, row + 1, col - (1 << (n - row - 2))));
                }

                if let Some(right) = node.borrow_mut().right.take() {
                    tmp.push((right, row + 1, col + (1 << (n - row - 2))));
                }
            }

            queue = tmp;
        }
    }

    grid
}

/// p657
pub fn judge_circle(moves: String) -> bool {
    let actions = moves.chars().collect::<Vec<char>>();
    const ORIGIN: [i32; 2] = [0, 0];
    let mut position = ORIGIN;
    actions.iter().for_each(|action| match action {
        'U' => {
            position[1] = position[1] + 1;
        }
        'D' => {
            position[1] = position[1] - 1;
        }
        'L' => {
            position[0] = position[0] - 1;
        }
        'R' => {
            position[0] = position[0] + 1;
        }
        _ => (),
    });
    position == ORIGIN
}

/// p658
pub fn find_closest_elements(arr: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
    let length = arr.len();
    let (mut left, mut right) = (0, length - 1);
    while left < right {
        let mid = (left + right) / 2;
        if arr[mid] == x {
            left = mid;
            break;
        } else if mid == left {
            left = if (x - arr[left]) <= (arr[right] - x) {
                mid
            } else {
                right
            };
            break;
        } else if arr[mid] < x {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    let k = k as usize;
    let mut start = 0;
    let mut end = length - 1;
    if left > k {
        start = left - k;
    }
    if length - left > k {
        end = left + k;
    }

    let mut ans = arr[start..=end].to_vec();
    while ans.len() > k {
        if (x - ans.first().unwrap()) <= (ans.last().unwrap() - x) {
            ans.pop();
        } else {
            ans.remove(0);
        }
    }
    ans
}

/// p659
pub fn is_possible(nums: Vec<i32>) -> bool {
    let mut map: HashMap<i32, BinaryHeap<Reverse<i32>>> = HashMap::new();
    for &num in nums.iter() {
        let cur_len = match map.get_mut(&(num - 1)) {
            Some(min_heap) if !min_heap.is_empty() => {
                let Reverse(prev_min_len) = min_heap.pop().unwrap();
                prev_min_len + 1
            }
            _ => 1,
        };
        map.entry(num)
            .or_insert(BinaryHeap::new())
            .push(Reverse(cur_len));
    }
    map.values().all(|min_heap| -> bool {
        match min_heap.peek() {
            Some(&Reverse(len)) if len < 3 => false,
            _ => true,
        }
    })
}

/// p661
pub fn image_smoother(img: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (r, c) = (img.len(), img[0].len());
    let mut ans = vec![vec![0; c]; r];
    for i in 0..r {
        for j in 0..c {
            let (mut cnt, mut sum) = (0, 0);
            for (x, y) in [
                (i - 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i, j - 1),
                (i, j),
                (i, j + 1),
                (i + 1, j - 1),
                (i + 1, j),
                (i + 1, j + 1),
            ] {
                if x < r && y < c {
                    cnt += 1;
                    sum += img[x][y];
                }
            }
            ans[i][j] = sum / cnt;
        }
    }
    ans
}

/// p662
pub fn width_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if let Some(node) = root {
        let mut nodes = vec![node.clone()];
        let mut positions = vec![1];
        let mut width = 1;
        fn bfs(
            nodes: &mut Vec<Rc<RefCell<TreeNode>>>,
            positions: &mut Vec<usize>,
            width: &mut usize,
        ) {
            if positions.len() == 0 {
                return;
            }
            if positions.len() >= 2 {
                *width = (*width).max(positions.last().unwrap() - positions.first().unwrap() + 1);
            }
            for _ in 0..nodes.len() {
                let node = nodes.remove(0);
                let position = positions.remove(0);
                if let Some(left) = node.borrow().left.clone() {
                    nodes.push(left);
                    positions.push(position * 2 - 1);
                }
                if let Some(right) = node.borrow().right.clone() {
                    nodes.push(right);
                    positions.push(position * 2);
                }
            }
            bfs(nodes, positions, width)
        }
        bfs(&mut nodes, &mut positions, &mut width);
        width as i32
    } else {
        0
    }
}

/// p664
pub fn strange_printer(s: String) -> i32 {
    let n = s.len();
    let s = s.as_bytes();
    let mut dp = vec![vec![0; n]; n];
    for i in (0..n).rev() {
        dp[i][i] = 1;
        for j in i + 1..n {
            dp[i][j] = if s[i] == s[j] {
                dp[i][j - 1]
            } else {
                let mut min_count = i32::MAX;
                for k in i..j {
                    min_count = i32::min(min_count, dp[i][k] + dp[k + 1][j]);
                }
                min_count
            };
        }
    }
    dp[0][n - 1]
}

/// p665
pub fn check_possibility(mut nums: Vec<i32>) -> bool {
    let n = nums.len();
    let mut flag = false;

    for i in 0..n - 1 {
        if nums[i] > nums[i + 1] {
            if flag {
                return false;
            }
            flag = true;
            if i > 0 && nums[i - 1] > nums[i + 1] {
                nums[i + 1] = nums[i];
            }
        }
    }

    true
}

/// p667
pub fn construct_array(n: i32, mut k: i32) -> Vec<i32> {
    let (mut l, mut r, mut ret) = (1, n, vec![0; n as usize]);
    for i in 0..n {
        let offset;
        if k % 2 == 0 {
            offset = r;
            r -= 1;
        } else {
            offset = l;
            l += 1;
        };
        ret[i as usize] = offset;
        if k > 1 {
            k -= 1;
        }
    }
    ret
}

/// p668
pub fn find_kth_number(m: i32, n: i32, k: i32) -> i32 {
    let (mut l, mut r) = (1, m * n + 1);
    while l < r {
        let (mid, mut cnt) = (l + ((r - l) >> 1), 0);
        for i in 1..=m {
            cnt += n.min(mid / i);
        }
        if cnt >= k {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    r
}

/// p669
pub fn trim_bst(
    root: Option<Rc<RefCell<TreeNode>>>,
    low: i32,
    high: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn dfs(
        root: &Option<Rc<RefCell<TreeNode>>>,
        low: i32,
        high: i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(node) = root.as_ref() {
            let v = node.borrow().val;
            let right_node = dfs(&node.borrow().right, low, high);
            let left_node = dfs(&node.borrow().left, low, high);
            if v < low {
                return right_node;
            } else if v > high {
                return left_node;
            } else {
                let new_node = Rc::new(RefCell::new(TreeNode::new(v)));
                new_node.borrow_mut().left = left_node;
                new_node.borrow_mut().right = right_node;
                Some(new_node)
            }
        } else {
            None
        }
    }
    dfs(&root, low, high)
}

/// p670
pub fn maximum_swap(num: i32) -> i32 {
    let mut s = num.to_string().into_bytes();
    let n = s.len();
    let mut max_idx = n - 1;
    let mut p = n;
    let mut q = 0;
    for i in (0..n - 1).rev() {
        if s[i] > s[max_idx] {
            max_idx = i;
        } else if s[i] < s[max_idx] {
            p = i;
            q = max_idx;
        }
    }
    if p == n {
        return num;
    }
    s.swap(p, q);
    unsafe { String::from_utf8_unchecked(s).parse().unwrap() }
}

/// p671
pub fn find_second_minimum_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut first = -1;
    let mut second = -1;
    fn dfs(root: Rc<RefCell<TreeNode>>, first: &mut i32, second: &mut i32) {
        let val = root.borrow().val;
        if *first == -1 || *first == val {
            *first = val;
            if let Some(left) = root.borrow_mut().left.take() {
                dfs(left, first, second);
            }
            if let Some(right) = root.borrow_mut().right.take() {
                dfs(right, first, second);
            }
        } else if *second == -1 || *second > val {
            *second = val;
        }
    }

    dfs(root.unwrap(), &mut first, &mut second);

    second
}

/// p672
pub fn flip_lights(n: i32, presses: i32) -> i32 {
    match presses {
        0 => 1,
        1 => {
            if n == 2 {
                3
            } else {
                (1 << n.min(4)).min(4)
            }
        }
        2 => (1 << n.min(4)).min(7),
        _ => (1 << n.min(4)).min(8),
    }
}

/// p673
pub fn find_number_of_lis(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut max = -1;
    let mut dp = vec![1; n];
    let mut count = vec![1; n];
    for i in 0..n {
        for j in 0..i {
            if nums[i] > nums[j] {
                if dp[i] < dp[j] + 1 {
                    dp[i] = dp[j] + 1;
                    count[i] = count[j];
                } else if dp[i] == dp[j] + 1 {
                    count[i] += count[j];
                }
            }
        }
        if max < dp[i] {
            max = dp[i];
        }
    }
    let mut ret = 0;
    for i in 0..n {
        if dp[i] == max {
            ret += count[i];
        }
    }
    ret
}

/// p674
pub fn find_length_of_lcis(nums: Vec<i32>) -> i32 {
    let mut ans = 1;
    let mut len = 1;
    nums.windows(2).for_each(|x| {
        if x[1] > x[0] {
            len += 1;
        } else {
            ans = ans.max(len);
            len = 1;
        }
    });
    ans.max(len)
}

/// p675
pub fn cut_off_tree(forest: Vec<Vec<i32>>) -> i32 {
    let (m, n, mut ret) = (forest.len(), forest[0].len(), 0);
    let mut cache = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if forest[i][j] > 1 {
                cache.push(vec![forest[i][j], i as i32, j as i32]);
            }
        }
    }
    let bfs = |i1: i32, j1: i32, i2: i32, j2: i32| -> i32 {
        use ::std::collections::VecDeque;
        if i1 == i2 && j1 == j2 {
            return 0;
        }
        let mut step = 0;
        let (mut queue, mut visited) = (VecDeque::new(), vec![vec![false; n]; m]);
        queue.push_back(vec![i1, j1]);
        visited[i1 as usize][j1 as usize] = true;
        while !queue.is_empty() {
            step += 1;
            let mut size = queue.len();
            while size > 0 {
                size -= 1;
                let curr = queue.pop_front().unwrap_or(Vec::new());
                let (curr_i, curr_j) = (curr[0], curr[1]);
                if curr_i == i2 && curr_j == j2 {
                    return step;
                }
                for dir in [[0, 1], [0, -1], [1, 0], [-1, 0]] {
                    let (x, y) = (curr_i + dir[0], curr_j + dir[1]);
                    if x < 0
                        || y < 0
                        || x >= m as i32
                        || y >= n as i32
                        || forest[x as usize][y as usize] == 0
                        || visited[x as usize][y as usize]
                    {
                        continue;
                    }
                    if x == i2 && y == j2 {
                        return step;
                    }
                    visited[x as usize][y as usize] = true;
                    queue.push_back(vec![x, y]);
                }
            }
        }
        -1
    };

    cache.sort_by(|a, b| a[0].cmp(&b[0]));
    let (mut i, mut j) = (0, 0);
    for c in cache {
        let step = bfs(i, j, c[1], c[2]);
        if step == -1 {
            return -1;
        }
        ret += step;
        i = c[1];
        j = c[2];
    }
    ret
}

/// p676
struct MagicDictionary {
    contents: RefCell<HashMap<usize, HashSet<String>>>,
}

impl MagicDictionary {
    fn new() -> Self {
        MagicDictionary {
            contents: RefCell::new(HashMap::new()),
        }
    }

    fn build_dict(&self, dictionary: Vec<String>) {
        let mut contents = self.contents.borrow_mut();
        for word in dictionary {
            let length = word.len();
            if let Some(words) = contents.get_mut(&length) {
                words.insert(word.clone());
            } else {
                let mut words = HashSet::new();
                words.insert(word.clone());
                contents.insert(length, words);
            }
        }
    }

    fn search(&self, search_word: String) -> bool {
        let length = search_word.len();
        let search_chars = search_word.chars().collect::<Vec<char>>();
        if let Some(words) = self.contents.borrow().get(&length) {
            let mut has_target = false;
            words.iter().for_each(|word| {
                let chars = word.chars().collect::<Vec<char>>();
                let mut differ_count = 0;
                for i in 0..length {
                    if chars[i] != search_chars[i] {
                        differ_count = differ_count + 1;
                    }
                }
                if differ_count == 1 {
                    has_target = true;
                }
            });
            return has_target;
        } else {
            return false;
        }
    }
}

/// p677
struct MapSum {
    contents: RefCell<HashMap<String, i32>>,
}

impl MapSum {
    fn new() -> Self {
        MapSum {
            contents: RefCell::new(HashMap::new()),
        }
    }

    fn insert(&self, key: String, val: i32) {
        self.contents.borrow_mut().insert(key, val);
    }

    fn sum(&self, prefix: String) -> i32 {
        self.contents.borrow().iter().fold(0, |acc, (word, val)| {
            if word.starts_with(&prefix) {
                return acc + val;
            } else {
                acc
            }
        })
    }
}

/// p678
pub fn check_valid_string(s: String) -> bool {
    let mut left_stack: Vec<usize> = Vec::new();
    let mut asterisk_stack: Vec<usize> = Vec::new();

    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' => {
                left_stack.push(i);
            }
            b')' => {
                if !left_stack.is_empty() {
                    left_stack.pop();
                } else if !asterisk_stack.is_empty() {
                    asterisk_stack.pop();
                } else {
                    return false;
                }
            }
            b'*' => {
                asterisk_stack.push(i);
            }
            _ => panic!("input error"),
        }
    }

    while !left_stack.is_empty() && !asterisk_stack.is_empty() {
        let left_idx = left_stack.pop().unwrap();
        let asterisk_idx = asterisk_stack.pop().unwrap();
        if left_idx >= asterisk_idx {
            return false;
        }
    }

    left_stack.is_empty()
}

/// p679
pub fn judge_point24(cards: Vec<i32>) -> bool {
    const EPS: f64 = 1e-9;

    fn dfs(cards: Vec<f64>) -> bool {
        let n = cards.len();
        if n == 1 {
            return (cards[0] - 24.0).abs() < EPS;
        }

        for (i, x) in cards.iter().enumerate() {
            for j in i + 1..n {
                let y = cards[j];

                let mut candidates = vec![x + y, x - y, y - x, x * y];
                if y.abs() > EPS {
                    candidates.push(x / y);
                }
                if x.abs() > EPS {
                    candidates.push(y / x);
                }

                for res in candidates {
                    let mut new_cards = cards.clone();
                    new_cards.remove(j);
                    new_cards[i] = res;
                    if dfs(new_cards) {
                        return true;
                    }
                }
            }
        }
        false
    }

    let a = cards.into_iter().map(|x| x as f64).collect::<Vec<_>>();
    dfs(a)
}

/// p680
pub fn valid_palindrome(s: String) -> bool {
    fn is_palindrome(s: &str, low: usize, high: usize) -> bool {
        let (mut low, mut high) = (low, high);
        let chars = s.chars().collect::<Vec<char>>();
        while low < high {
            if chars[low] != chars[high] {
                return false;
            } else {
                low = low + 1;
                high = high - 1;
            }
        }
        true
    }
    let chars = s.chars().collect::<Vec<char>>();
    let (mut low, mut high) = (0, chars.len() - 1);
    while low < high {
        if chars[low] == chars[high] {
            low = low + 1;
            high = high - 1;
        } else {
            return is_palindrome(&s, low, high - 1) || is_palindrome(&s, low + 1, high);
        }
    }
    true
}

/// p682
pub fn cal_points(operations: Vec<String>) -> i32 {
    let mut points = Vec::new();
    operations.iter().for_each(|op| match op {
        d if d == &"D".to_string() => {
            if let Some(v) = points.last() {
                points.push(v * 2);
            }
        }
        c if c == &"C".to_string() => {
            points.pop();
        }
        add if add == &"+".to_string() => {
            let length = points.len();
            points.push(points[length - 1] + points[length - 2]);
        }
        num => {
            if let Ok(v) = num.parse::<i32>() {
                points.push(v)
            }
        }
    });
    points.iter().fold(0, |acc, cur| acc + cur)
}

/// p684
pub fn find_redundant_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
    let mut parent: Vec<_> = (0..=1000).collect();
    let mut ans = vec![];

    fn union(parent: &mut Vec<usize>, idx1: usize, idx2: usize) {
        let idx1 = find(parent, idx1);
        let idx2 = find(parent, idx2);
        parent[idx1] = idx2;
    }
    fn find(parent: &mut Vec<usize>, mut idx: usize) -> usize {
        while parent[idx] != idx {
            parent[idx] = parent[parent[idx]];
            idx = parent[idx];
        }
        idx
    }
    for edge in edges.iter() {
        if find(&mut parent, edge[0] as usize) == find(&mut parent, edge[1] as usize) {
            ans = edge.clone();
            break;
        } else {
            union(&mut parent, edge[0] as usize, edge[1] as usize);
        }
    }
    ans
}

/// p685
pub fn find_redundant_directed_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
    let n = edges.len();
    let mut father: Vec<i32> = Vec::with_capacity(n + 1);
    let mut union_set: Vec<usize> = Vec::with_capacity(n + 1);

    let mut triangle: Option<(i32, i32, i32)> = None;
    let mut cycle: Option<(i32, i32)> = None;

    for i in 0..=n {
        father.push(i as i32);
        union_set.push(i);
    }
    for e in edges.iter() {
        let r = e[0];
        let t = e[1];
        let ru = r as usize;
        let tu = t as usize;

        if father[tu] != t {
            triangle = Some((father[tu], r, t));
        } else {
            father[tu] = r;

            let mut rx = ru;
            let mut tx = tu;
            while union_set[rx] != rx {
                rx = union_set[rx];
            }
            while union_set[tx] != tx {
                tx = union_set[tx];
            }

            if rx == tx {
                cycle = Some((r, t));
            } else {
                union_set[tx] = rx;
            }
        }
    }
    return if let Some((a, b, c)) = triangle {
        if let Some(_) = cycle {
            vec![a, c]
        } else {
            vec![b, c]
        }
    } else {
        if let Some((r, t)) = cycle {
            vec![r, t]
        } else {
            panic!()
        }
    };
}

/// p686
pub fn repeated_string_match(a: String, b: String) -> i32 {
    let m = a.len();
    let n = b.len();
    let ret = n / m;
    let mut exist = vec![false; 26];

    a.chars()
        .for_each(|ch| exist[(ch as i32 - 'a' as i32) as usize] = true);
    for ch in b.chars() {
        if !exist[(ch as i32 - 'a' as i32) as usize] {
            return -1;
        }
    }

    let mut s = a.repeat(ret);
    for i in 0..3 {
        if s.contains(&b) {
            return ret as i32 + i;
        }
        s += &a;
    }
    -1
}

/// p687
pub fn longest_univalue_path(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }
    let mut maximum = 0;
    fn recursive(node: &Option<Rc<RefCell<TreeNode>>>, val: i32, maximum: &mut i32) -> i32 {
        if node.is_none() {
            return 0;
        }
        let (l, r) = (
            recursive(
                &node.as_ref().unwrap().borrow().left,
                node.as_ref().unwrap().borrow().val,
                maximum,
            ),
            recursive(
                &node.as_ref().unwrap().borrow().right,
                node.as_ref().unwrap().borrow().val,
                maximum,
            ),
        );
        *maximum = *maximum.max(&mut (l + r));
        if node.as_ref().unwrap().borrow().val == val {
            l.max(r) + 1
        } else {
            0
        }
    }
    recursive(&root, root.as_ref().unwrap().borrow().val, &mut maximum);
    maximum
}

/// p688
pub fn knight_probability(n: i32, k: i32, row: i32, column: i32) -> f64 {
    const DIRS: [[i32; 2]; 8] = [
        [-2, -1],
        [-2, 1],
        [2, -1],
        [2, 1],
        [-1, -2],
        [-1, 2],
        [1, -2],
        [1, 2],
    ];
    let n = n as usize;
    let k = k as usize;
    let mut dp = vec![vec![vec![0.0; n]; n]; k + 1];

    for step in 0..=k {
        for i in 0..n {
            for j in 0..n {
                if step == 0 {
                    dp[step][i][j] = 1.0;
                } else {
                    for dir in &DIRS {
                        let ni = i as i32 + dir[0];
                        let nj = j as i32 + dir[1];
                        if ni >= 0 && ni < n as i32 && nj >= 0 && nj < n as i32 {
                            dp[step][i][j] += dp[step - 1][ni as usize][nj as usize] / 8.0;
                        }
                    }
                }
            }
        }
    }

    dp[k][row as usize][column as usize]
}

/// p689
pub fn max_sum_of_three_subarrays(nums: Vec<i32>, k: i32) -> Vec<i32> {
    let (n, k) = (nums.len(), k as usize);
    let (mut sum1, mut sum2, mut sum3) = (0, 0, 0);
    let (mut max_sum1, mut max_sum12, mut max_sum123) = (0, 0, 0);
    let (mut idx1, mut idx2, mut idx3) = (-1, -1, -1);
    let mut max_idx1 = -1;
    let mut res = Vec::new();

    for i in 2 * k..n {
        sum1 += nums[i - 2 * k];
        sum2 += nums[i - k];
        sum3 += nums[i];

        if i + 1 >= 3 * k {
            if sum1 > max_sum1 {
                max_sum1 = sum1;
                idx1 = (i - 3 * k + 1) as i32;
            }
            if max_sum1 + sum2 > max_sum12 {
                max_sum12 = max_sum1 + sum2;
                max_idx1 = idx1;
                idx2 = (i - 2 * k + 1) as i32;
            }
            if max_sum12 + sum3 > max_sum123 {
                max_sum123 = max_sum12 + sum3;
                idx3 = (i - k + 1) as i32;
                res = vec![max_idx1, idx2, idx3];
            }
            sum1 -= nums[i - 3 * k + 1];
            sum2 -= nums[i - 2 * k + 1];
            sum3 -= nums[i - k + 1];
        }
    }

    res
}

/// p691
pub fn min_stickers(stickers: Vec<String>, target: String) -> i32 {
    fn remove_dominated_stickers(stickers_cnt: &mut Vec<Vec<i32>>) -> i32 {
        let (mut start, m, n) = (0, stickers_cnt.len(), stickers_cnt[0].len());
        for i in 0..m {
            for j in start..m {
                if j != i {
                    let mut k = 0;
                    while k < n && stickers_cnt[i][k] <= stickers_cnt[j][k] {
                        k += 1;
                    }
                    if k == n {
                        stickers_cnt.swap(i, start);
                        start += 1;
                        break;
                    }
                }
            }
        }
        start as i32
    }

    fn to_string(freq: &Vec<i32>) -> String {
        let (mut s, mut ch) = (String::new(), 'a');
        for f in freq {
            let mut k = *f;
            while k > 0 {
                k -= 1;
                s.push(ch)
            }
            ch = (ch as u8 + 1) as char
        }
        s
    }

    let mut target_native_cnt = vec![0; 26];
    target
        .chars()
        .for_each(|ch| target_native_cnt[(ch as u8 - 'a' as u8) as usize] += 1);
    let (mut index, mut n): (Vec<i32>, _) = (vec![0; 26], 0);
    for i in 0..26 {
        if target_native_cnt[i as usize] > 0 {
            index[i as usize] = n;
            n += 1
        } else {
            index[i as usize] = -1;
        }
    }
    let (mut target_cnt, mut t) = (vec![0; n as usize], 0);
    target_native_cnt.iter().for_each(|c| {
        if *c > 0 {
            target_cnt[t] = *c;
            t += 1;
        }
    });
    let m = stickers.len();
    let mut stickers_cnt = vec![vec![0; n as usize]; m];
    for i in 0..m {
        for ch in stickers[i].chars() {
            let j = index[(ch as u8 - 'a' as u8) as usize];
            if j >= 0 {
                stickers_cnt[i][j as usize] += 1;
            }
        }
    }
    let start = remove_dominated_stickers(&mut stickers_cnt);
    let (mut queue, mut visited, mut step) = (VecDeque::new(), HashSet::new(), 0);
    queue.push_back(target_cnt);
    while !queue.is_empty() {
        step += 1;
        let mut size = queue.len();
        while size > 0 {
            size -= 1;
            let freq = queue.pop_front().unwrap_or(vec![]);
            let curr = to_string(&freq);
            if visited.contains(&curr) {
                continue;
            }
            let first = (curr.chars().collect::<Vec<_>>()[0] as u8 - 'a' as u8) as usize;
            visited.insert(curr);
            for i in start as usize..stickers.len() {
                if stickers_cnt[i][first] > 0 {
                    let (mut next, mut k) = (freq.clone(), 0);
                    for j in 0..n {
                        next[j as usize] = (next[j as usize] - stickers_cnt[i][j as usize]).max(0);
                        if next[j as usize] == 0 {
                            k += 1;
                            if k == n {
                                return step;
                            }
                        }
                    }
                    queue.push_back(next);
                }
            }
        }
    }
    -1
}

/// p692
pub fn top_k_frequent(words: Vec<String>, k: i32) -> Vec<String> {
    let mut map = HashMap::<String, usize>::new();
    words
        .into_iter()
        .for_each(|word| *map.entry(word).or_default() += 1);
    map.into_iter()
        .map(|(w, n)| (Reverse(n), w))
        .collect::<BinaryHeap<_>>()
        .into_sorted_vec()
        .into_iter()
        .map(|(_, w)| w)
        .take(k as usize)
        .collect()
}

///p693
pub fn has_alternating_bits(n: i32) -> bool {
    let mut n = n;
    let mut prv = 1 - n % 2;
    while n > 0 {
        if prv == n % 2 {
            return false;
        }
        prv = n % 2;
        n /= 2;
    }
    true
}

/// p695
pub fn max_area_of_island(grid: Vec<Vec<i32>>) -> i32 {
    const DIRECTIONS: [[i32; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    let mut grid = grid;
    let mut max = 0;
    let width = grid.len();
    let length = grid[0].len();
    fn dfs(grid: &mut Vec<Vec<i32>>, width: usize, length: usize, i: i32, j: i32) -> i32 {
        if i < 0
            || j < 0
            || i as usize == width
            || j as usize == length
            || grid[i as usize][j as usize] == 0
        {
            return 0;
        } else {
            grid[i as usize][j as usize] = 0;
            let mut acc = 1;
            for direction in DIRECTIONS {
                acc = acc + dfs(grid, width, length, i + direction[0], j + direction[1])
            }
            acc
        }
    }
    for i in 0..width {
        for j in 0..length {
            max = max.max(dfs(&mut grid, width, length, i as i32, j as i32));
        }
    }
    max
}
