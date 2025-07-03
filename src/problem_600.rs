use std::cell::RefCell;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::rc::Rc;

use crate::common::TreeNode;

/// p501
pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut frequency: HashMap<i32, i32> = HashMap::new();
    fn dfs(root: Rc<RefCell<TreeNode>>, frequncy: &mut HashMap<i32, i32>) {
        let count = frequncy.entry(root.borrow().val).or_insert(0);
        *count = *count + 1;
        if let Some(left) = root.clone().borrow().left.as_ref() {
            dfs(left.clone(), frequncy);
        }
        if let Some(right) = root.borrow().right.as_ref() {
            dfs(right.clone(), frequncy);
        }
    }
    if let Some(val) = root {
        dfs(val, &mut frequency);
    }
    let mut max: i32 = 0;
    let mut keys: Vec<i32> = vec![];
    frequency.iter().for_each(|(key, val)| {
        if *val > max {
            keys.clear();
            keys.push(*key);
            max = *val;
        } else if *val == max {
            keys.push(*key)
        }
    });
    keys
}

/// p502
pub fn find_maximized_capital(k: i32, w: i32, profits: Vec<i32>, capital: Vec<i32>) -> i32 {
    let mut map: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for (key, val) in capital.into_iter().zip(profits.into_iter()) {
        map.entry(key).or_insert(Vec::new()).push(val);
    }
    let mut max_heap: BinaryHeap<i32> = BinaryHeap::new();
    insert(&map, -1, w, &mut max_heap);

    let mut cur_profit = w;
    let mut count = 0;
    while count < k {
        match max_heap.pop() {
            Some(val) if val > 0 => {
                cur_profit += val;
                insert(&map, cur_profit - val, cur_profit, &mut max_heap);
                count += 1;
            }
            _ => break,
        }
    }

    cur_profit
}

fn insert(map: &BTreeMap<i32, Vec<i32>>, left: i32, right: i32, max_heap: &mut BinaryHeap<i32>) {
    if left + 1 > right {
        return;
    }
    for (_, vals) in map.range(left + 1..=right) {
        for &v in vals.iter() {
            max_heap.push(v);
        }
    }
}

/// p503
pub fn next_greater_elements(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut ans = vec![-1; n];
    let mut st = vec![];
    for i in (0..2 * n).rev() {
        let x = nums[i % n];
        while let Some(&top) = st.last() {
            if x < top {
                break;
            }
            st.pop();
        }
        if i < n && !st.is_empty() {
            ans[i] = *st.last().unwrap();
        }
        st.push(x);
    }
    ans
}

/// p504
pub fn convert_to_base7(num: i32) -> String {
    if num == 0 {
        return "0".to_string();
    }
    let mut symbols: Vec<char> = vec![];
    let mut remain = num.abs();

    while remain != 0 {
        symbols.push((remain % 7).to_string().chars().collect::<Vec<char>>()[0]);
        remain = remain / 7;
    }
    if num < 0 {
        symbols.push('-');
    }
    symbols.reverse();
    symbols.iter().collect::<String>()
}
