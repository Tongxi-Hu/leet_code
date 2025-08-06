use std::{cell::RefCell, collections::HashMap, rc::Rc};

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
