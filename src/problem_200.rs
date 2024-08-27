use std::{cell::RefCell, rc::Rc};

use crate::common::TreeNode;

///p101
///
/// Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn check(left: &Option<Rc<RefCell<TreeNode>>>, right: &Option<Rc<RefCell<TreeNode>>>) -> bool {
        match (left, right) {
            (None, None) => true,
            (Some(_), None) => false,
            (None, Some(_)) => false,
            (Some(left_node), Some(right_node)) => {
                return left_node.borrow().val == right_node.borrow().val
                    && check(&left_node.borrow().left, &right_node.borrow().right)
                    && check(&left_node.borrow().right, &right_node.borrow().left);
            }
        }
    }
    match root {
        None => return true,
        Some(root_node) => return check(&root_node.borrow().left, &root_node.borrow().right),
    }
}

/// p102
///
/// Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut val: Vec<Vec<i32>> = vec![];
    let mut current_level: Vec<Rc<RefCell<TreeNode>>> = vec![];
    if let Some(node) = root {
        current_level.push(node);
    }
    while !current_level.is_empty() {
        let mut next = vec![];
        let mut values = vec![];
        for cur in current_level {
            let mut node = cur.borrow_mut();
            values.push(node.val);
            if let Some(left) = node.left.take() {
                next.push(left);
            }
            if let Some(right) = node.right.take() {
                next.push(right);
            }
        }
        val.push(values);
        current_level = next;
    }
    return val;
}

///p103
///
///Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).
pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut current_level: Vec<Rc<RefCell<TreeNode>>> = vec![];
    let mut from_left = true;
    if let Some(val) = root {
        current_level.push(val);
    }
    while !current_level.is_empty() {
        let mut next = vec![];
        let mut values = vec![];
        for item in current_level {
            let mut node = item.borrow_mut();
            values.push(node.val);
            if let Some(left) = node.left.take() {
                next.push(left)
            }
            if let Some(right) = node.right.take() {
                next.push(right)
            }
        }
        current_level = next;
        if from_left {
            ans.push(values);
        } else {
            values.reverse();
            ans.push(values);
        }
        from_left = !from_left;
    }
    return ans;
}

///p104
///
///A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut depth = 0;
    if let Some(val) = root {
        depth = 1;
        let mut node = val.borrow_mut();
        depth = depth + max_depth(node.left.take()).max(max_depth(node.right.take()));
    }
    return depth;
}
