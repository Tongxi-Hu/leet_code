use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashMap},
    rc::Rc,
};

use crate::{common::TreeNode, problem_200::largest_number};

/// p401
fn read_binary_watch(turned_on: i32) -> Vec<String> {
    let mut res = Vec::new();
    for h in 0..12 as i32 {
        for m in 0..60 as i32 {
            if h.count_ones() + m.count_ones() == (turned_on as u32) {
                res.push(format!("{}:{:02}", h, m))
            }
        }
    }
    res
}

/// p402
pub fn remove_kdigits(num: String, mut k: i32) -> String {
    let mut stack = vec![];

    for b in num.bytes() {
        while k > 0 {
            match stack.last() {
                Some(&prev_b) if prev_b > b => {
                    stack.pop();
                    k -= 1;
                }
                _ => break,
            }
        }

        if !stack.is_empty() || b != b'0' {
            stack.push(b);
        }
    }

    let n = stack.len();
    let k = k as usize;
    if n > k {
        String::from_utf8_lossy(&stack[..n - k]).to_string()
    } else {
        "0".to_string()
    }
}

/// p403
pub fn can_cross(stones: Vec<i32>) -> bool {
    let mut maps: HashMap<(usize, i32), bool> = HashMap::new();
    fn dfs(
        stones: &Vec<i32>,
        maps: &mut HashMap<(usize, i32), bool>,
        cur: usize,
        last_distance: i32,
    ) -> bool {
        if cur + 1 == stones.len() {
            return true;
        }
        if let Some(&flag) = maps.get(&(cur, last_distance)) {
            return flag;
        }

        for next_distance in last_distance - 1..=last_distance + 1 {
            if next_distance <= 0 {
                continue;
            }
            let next = stones[cur] + next_distance;
            if let Ok(next_idx) = stones.binary_search(&next) {
                if dfs(stones, maps, next_idx, next_distance) {
                    maps.insert((cur, last_distance), true);
                    return true;
                }
            }
        }

        maps.insert((cur, last_distance), false);
        false
    }
    dfs(&stones, &mut maps, 0, 0)
}

/// p404
pub fn sum_of_left_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut res = 0;
    fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, is_left: bool, res: &mut i32) {
        if let Some(content) = root {
            let left_node = &content.borrow().left;
            let right_node = &content.borrow().right;
            if *left_node == None && *right_node == None {
                if is_left {
                    *res = *res + content.borrow().val
                }
            } else {
                dfs(left_node, true, res);
                dfs(right_node, false, res);
            }
        }
    }
    dfs(&root, false, &mut res);
    res
}

/// p405
pub fn to_hex(num: i32) -> String {
    format!("{:x}", num)
}

/// p406
pub fn reconstruct_queue(people: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut people = people;
    let mut ans = vec![];
    people.sort_by(|a, b| b[0].cmp(&a[0]).then(a[1].cmp(&b[1])));
    for p in people.iter() {
        ans.insert(p[1] as usize, p.to_vec())
    }
    ans
}

/// p407
pub fn trap_rain_water(mut height_map: Vec<Vec<i32>>) -> i32 {
    let m = height_map.len();
    let n = height_map[0].len();
    let mut h = BinaryHeap::new();
    for (i, row) in height_map.iter_mut().enumerate() {
        for (j, height) in row.iter_mut().enumerate() {
            if i == 0 || i == m - 1 || j == 0 || j == n - 1 {
                h.push((-*height, i, j)); // 取相反数变成最小堆
                *height = -1; // 标记 (i,j) 访问过
            }
        }
    }

    let mut ans = 0;
    while let Some((min_height, i, j)) = h.pop() {
        // 去掉短板
        let min_height = -min_height; // min_height 是木桶的短板
        for (x, y) in [(i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j)] {
            if x < m && y < n && height_map[x][y] >= 0 {
                // (x,y) 没有访问过
                // 如果 (x,y) 的高度小于 min_height，那么接水量为 min_height - heightMap[x][y]
                ans += 0.max(min_height - height_map[x][y]);
                // 给木桶新增一块高为 max(min_height, heightMap[x][y]) 的木板
                h.push((-min_height.max(height_map[x][y]), x, y));
                height_map[x][y] = -1; // 标记 (x,y) 访问过
            }
        }
    }
    ans
}

/// p409
pub fn longest_palindrome(s: String) -> i32 {
    let chars = s.chars();
    let mut record: HashMap<char, usize> = HashMap::new();
    for c in chars {
        if let Some(count) = record.get_mut(&c) {
            *count = *count + 1;
        } else {
            record.insert(c, 1);
        }
    }
    let mut length = 0;
    for l in record.values() {
        length = length + l / 2 * 2;
        if l % 2 == 1 && length % 2 == 0 {
            length = length + 1
        }
    }
    length as i32
}

/// p410
pub fn split_array(nums: Vec<i32>, k: i32) -> i32 {
    let check = |mx: i32| -> bool {
        let mut cnt = 1;
        let mut s = 0;
        for &x in &nums {
            if s + x <= mx {
                s += x;
            } else {
                if cnt == k {
                    return false;
                }
                cnt += 1;
                s = x;
            }
        }
        true
    };

    let mut right = nums.iter().sum::<i32>();
    let mut left = (*nums.iter().max().unwrap() - 1).max((right - 1) / k);
    while left + 1 < right {
        let mid = left + (right - left) / 2;
        if check(mid) {
            right = mid;
        } else {
            left = mid;
        }
    }
    right
}

/// p412
pub fn fizz_buzz(n: i32) -> Vec<String> {
    let mut ans: Vec<String> = vec![];
    for i in 1..=n {
        if i % 15 == 0 {
            ans.push("FizzBuzz".to_string());
        } else if i % 5 == 0 {
            ans.push("Buzz".to_string());
        } else if i % 3 == 0 {
            ans.push("Fizz".to_string())
        } else {
            ans.push(i.to_string())
        }
    }
    ans
}

/// p413
pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    if n == 1 {
        return 0;
    }
    let (mut d, mut t, mut ans) = (nums[0] - nums[1], 0, 0);
    for i in 2..n {
        if nums[i - 1] - nums[i] == d {
            t = t + 1;
        } else {
            d = nums[i - 1] - nums[i];
            t = 0;
        }
        ans = ans + t;
    }
    return ans;
}

/// p414
pub fn third_max(nums: Vec<i32>) -> i32 {
    let (mut first, mut second, mut third) = (i64::MIN, i64::MIN, i64::MIN);
    for n in nums {
        let num = n as i64;
        if num > first {
            third = second;
            second = first;
            first = num;
        } else if num < first && num > second {
            third = second;
            second = num;
        } else if num < second && num > third {
            third = num;
        }
    }
    return if third == i64::MIN {
        first as i32
    } else {
        third as i32
    };
}

/// p415
pub fn add_strings(num1: String, num2: String) -> String {
    use std::iter::repeat;
    if num2.len() > num1.len() {
        return add_strings(num2, num1);
    }
    let mut prev = 0;
    let mut ret = num1
        .chars()
        .rev()
        .zip(
            num2.chars()
                .rev()
                .chain(repeat('0').take(num1.len().saturating_sub(num2.len()))),
        )
        .map(|(a, b)| {
            let curr = prev + a.to_digit(10).unwrap() + b.to_digit(10).unwrap();
            prev = curr / 10;
            char::from_digit(curr % 10, 10).unwrap()
        })
        .collect::<Vec<_>>();

    if prev == 1 {
        ret.push((1u8 + b'0') as char);
    }
    ret.iter().rev().collect::<_>()
}
