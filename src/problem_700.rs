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
