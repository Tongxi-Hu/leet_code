use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::common::TreeNode;

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
