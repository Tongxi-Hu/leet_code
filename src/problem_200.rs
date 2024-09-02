use std::{cell::RefCell, rc::Rc};

use crate::common::{ListNode, TreeNode};

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

///p105
///
///Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.
pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    if preorder.is_empty() {
        return None;
    }
    if let Some(index) = inorder.iter().position(|item| *item == preorder[0]) {
        let left_pre = preorder[1..1 + index].to_vec();
        let right_pre = preorder[1 + index..].to_vec();
        let left_in = inorder[..index].to_vec();
        let right_in = inorder[index + 1..].to_vec();
        let left_node = build_tree(left_pre, left_in);
        let right_node = build_tree(right_pre, right_in);
        return Some(Rc::new(RefCell::new(TreeNode {
            val: preorder[0],
            left: left_node,
            right: right_node,
        })));
    }
    return None;
}

///p106
///
/// Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree.
pub fn build_tree_2(inorder: Vec<i32>, postorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    if inorder.is_empty() {
        return None;
    }
    if let Some(index) = inorder
        .iter()
        .position(|item| item == postorder.last().unwrap())
    {
        let left_post = postorder[0..index].to_vec();
        let right_post = postorder[index..postorder.len() - 1].to_vec();
        let left_in = inorder[..index].to_vec();
        let right_in = inorder[index + 1..].to_vec();
        let left_node = build_tree_2(left_in, left_post);
        let right_node = build_tree_2(right_in, right_post);
        return Some(Rc::new(RefCell::new(TreeNode {
            val: *postorder.last().unwrap(),
            left: left_node,
            right: right_node,
        })));
    }
    return None;
}

///p107
///
///Given the root of a binary tree, return the bottom-up level order traversal of its nodes' values. (i.e., from left to right, level by level from leaf to root).
pub fn level_order_bottom(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
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
    val.reverse();
    return val;
}

///p108
///
/// Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.
pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    if nums.is_empty() {
        return None;
    }
    let mid = (nums.len() - 1) / 2;
    let left = sorted_array_to_bst(nums[0..mid].to_vec());
    let right = sorted_array_to_bst(nums[mid + 1..].to_vec());
    return Some(Rc::new(RefCell::new(TreeNode {
        val: nums[mid],
        left,
        right,
    })));
}

///p109
///
///Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height-balanced binary search tree.
pub fn sorted_list_to_bst(head: Option<Box<ListNode>>) -> Option<Rc<RefCell<TreeNode>>> {
    let mut head = head;
    let mut nums: Vec<i32> = vec![];
    while let Some(mut node) = head {
        nums.push(node.val);
        head = node.next.take();
    }
    fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        if nums.is_empty() {
            return None;
        }
        let mid = (nums.len() - 1) / 2;
        let left = sorted_array_to_bst(nums[0..mid].to_vec());
        let right = sorted_array_to_bst(nums[mid + 1..].to_vec());
        return Some(Rc::new(RefCell::new(TreeNode {
            val: nums[mid],
            left,
            right,
        })));
    }
    return sorted_array_to_bst(nums);
}

///p110
///
///Given a binary tree, determine if it is height-balanced.
pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn is_balanced_and_depth(root: Option<Rc<RefCell<TreeNode>>>) -> (bool, usize) {
        let root = root;
        match root {
            None => return (true, 0),
            Some(node) => {
                let left = is_balanced_and_depth(node.borrow_mut().left.take());
                let right = is_balanced_and_depth(node.borrow_mut().right.take());
                return (
                    left.0 && right.0 && left.1.abs_diff(right.1) <= 1,
                    left.1.max(right.1) + 1,
                );
            }
        };
    }
    return is_balanced_and_depth(root).0;
}

///p111
///
/// Given a binary tree, find its minimum depth.
///The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
pub fn min_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if let Some(node) = root {
        let mut node = node.borrow_mut();
        if node.right.is_none() {
            return min_depth(node.left.take()) + 1;
        }
        if node.left.is_none() {
            return min_depth(node.right.take()) + 1;
        }
        return min_depth(node.left.take()).min(min_depth(node.right.take())) + 1;
    }
    return 0;
}

///p112
///
///Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.
pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
    if let Some(node) = root {
        let mut node = node.borrow_mut();
        let target_sum = target_sum - node.val;
        if node.left.is_none() && node.right.is_none() {
            return target_sum == 0;
        }
        return has_path_sum(node.left.take(), target_sum)
            || has_path_sum(node.right.take(), target_sum);
    }
    return false;
}

///p113
///
/// Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references. A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.
pub fn path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> Vec<Vec<i32>> {
    let mut all_path: Vec<Vec<i32>> = vec![];
    let mut cur_path: Vec<i32> = vec![];
    fn dfs(
        root: &Option<Rc<RefCell<TreeNode>>>,
        cur_path: &mut Vec<i32>,
        all_path: &mut Vec<Vec<i32>>,
        target_sum: i32,
    ) {
        match root {
            None => return,
            Some(val) => {
                let node = val.borrow_mut();
                cur_path.push(node.val);
                if node.left.is_none() && node.right.is_none() {
                    let cur_sum = cur_path.iter().fold(0, |acc, cur| acc + *cur);
                    if cur_sum == target_sum {
                        all_path.push(cur_path.to_vec());
                    }
                } else {
                    dfs(&node.left, cur_path, all_path, target_sum);
                    dfs(&node.right, cur_path, all_path, target_sum);
                }
                cur_path.pop();
            }
        }
    }
    dfs(&root, &mut cur_path, &mut all_path, target_sum);
    return all_path;
}

///p114
///
/// Given the root of a binary tree, flatten the tree into a "linked list":
/// The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
/// The "linked list" should be in the same order as a pre-order traversal of the binary tree.
pub fn flatten(root: &mut Option<Rc<RefCell<TreeNode>>>) {
    let mut curr = root.as_ref().map(|n| n.clone());
    while let Some(curr_node) = curr {
        let mut curr_node = curr_node.borrow_mut();
        if let Some(next_node) = curr_node.left.take() {
            let mut predecessor = next_node.clone();
            let mut predecessor_right = predecessor.borrow().right.clone();
            while let Some(node) = predecessor_right {
                predecessor_right = node.borrow().right.clone();
                predecessor = node;
            }
            predecessor.borrow_mut().right = curr_node.right.take();
            curr_node.right = Some(next_node);
        }
        curr = curr_node.right.clone();
    }
}

///p115
///
/// Given two strings s and t, return the number of distinct subsequences of s which equals t. The test cases are generated so that the answer fits on a 32-bit signed integer.
pub fn num_distinct(s: String, t: String) -> i32 {
    let s_char = s.chars().collect::<Vec<char>>();
    let s_len = s_char.len();
    let t_char = t.chars().collect::<Vec<char>>();
    let t_len = t_char.len();
    if s_char.len() < t_char.len() {
        return 0;
    }
    let mut dp: Vec<Vec<usize>> = vec![vec![0; t_len + 1]; s_len + 1];
    for i in 0..=s_len {
        dp[i][t_len] = 1;
    }
    for i in (0..=s_len - 1).rev() {
        for j in (0..=t_len - 1).rev() {
            if s_char[i] == t_char[j] {
                dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j];
            } else {
                dp[i][j] = dp[i + 1][j];
            }
        }
    }
    return dp[0][0] as i32;
}

///p118
///
///Given an integer numRows, return the first numRows of Pascal's triangle.
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let num_rows = num_rows as usize;
    let mut ans: Vec<Vec<i32>> = vec![vec![1]];
    if num_rows == 1 {
        return ans;
    }
    ans.push(vec![1, 1]);
    if num_rows == 2 {
        return ans;
    }
    for i in 3..=num_rows {
        let mut temp = vec![0; i];
        let last = ans.last().unwrap();
        temp[0] = 1;
        temp[i - 1] = 1;
        for j in 1..=i - 2 {
            temp[j] = last[j - 1] + last[j]
        }
        ans.push(temp);
    }
    return ans;
}

///p119
///
/// Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.
pub fn get_row(row_index: i32) -> Vec<i32> {
    let row_index: usize = row_index as usize;
    if row_index == 0 {
        return vec![1];
    };
    if row_index == 1 {
        return vec![1, 1];
    }
    let last = get_row((row_index - 1) as i32);
    let mut ans = vec![0; row_index + 1];
    ans[0] = 1;
    ans[row_index] = 1;
    for i in 1..=row_index - 1 {
        ans[i] = last[i - 1] + last[i]
    }
    return ans;
}

///p120
///
///Given a triangle array, return the minimum path sum from top to bottom.
///For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.
pub fn minimum_total(triangle: Vec<Vec<i32>>) -> i32 {
    let mut min_sum: Vec<Vec<i32>> = vec![];
    min_sum.push(triangle[0].to_vec());
    for i in 1..triangle.len() {
        let mut temp = vec![];
        for j in 0..triangle[i].len() {
            if j == 0 {
                temp.push(triangle[i][j] + min_sum[i - 1][j]);
            } else if j == triangle[i].len() - 1 {
                temp.push(triangle[i][j] + min_sum[i - 1][j - 1]);
            } else {
                temp.push(triangle[i][j] + min_sum[i - 1][j - 1].min(min_sum[i - 1][j]));
            }
        }
        min_sum.push(temp.to_vec())
    }
    return min_sum
        .last()
        .unwrap()
        .into_iter()
        .fold(core::i32::MAX, |acc, cur| acc.min(*cur));
}
