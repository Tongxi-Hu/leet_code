use std::{cell::RefCell, num::NonZero, rc::Rc};

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

///p121
///
///You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Find and return the maximum profit you can achieve.
pub fn max_profit(prices: Vec<i32>) -> i32 {
    let mut min = core::i32::MAX;
    let mut max_profit = 0;
    for &price in prices.iter() {
        if price < min {
            min = price;
        } else {
            max_profit = max_profit.max(price - min);
        }
    }
    return max_profit;
}

///p122
///
/// You are given an integer array prices where prices[i] is the price of a given stock on the ith day. On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day. Find and return the maximum profit you can achieve.
pub fn max_profit_2(prices: Vec<i32>) -> i32 {
    let mut dp: Vec<(i32, i32)> = vec![];
    dp.push((0, -prices[0]));
    for i in 1..prices.len() {
        dp.push((
            dp[i - 1].0.max(dp[i - 1].1 + prices[i]),
            dp[i - 1].1.max(dp[i - 1].0 - prices[i]),
        ))
    }
    return dp[prices.len() - 1].0;
}

///p123
///
/// You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit you can achieve. You may complete at most two transactions. Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
pub fn max_profit_3(prices: Vec<i32>) -> i32 {
    let length = prices.len();
    let (mut buy_1, mut sell_1, mut buy_2, mut sell_2) = (-prices[0], 0, -prices[0], 0);
    for i in 0..length {
        buy_1 = buy_1.max(-prices[i]);
        sell_1 = sell_1.max(buy_1 + prices[i]);
        buy_2 = buy_2.max(sell_1 - prices[i]);
        sell_2 = sell_2.max(buy_2 + prices[i]);
    }
    return sell_2;
}

///p124
///
///A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root. The path sum of a path is the sum of the node's values in the path. Given the root of a binary tree, return the maximum path sum of any non-empty path.
pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut max_path = core::i32::MIN;
    fn max_gain(root: &Option<Rc<RefCell<TreeNode>>>, max_path: &mut i32) -> i32 {
        if let Some(node) = root {
            let left_gain = max_gain(&node.borrow().left, max_path).max(0);
            let right_gain = max_gain(&node.borrow().right, max_path).max(0);
            let new_path = node.borrow().val + left_gain + right_gain;
            *max_path = (*max_path).max(new_path);
            return node.borrow().val + left_gain.max(right_gain);
        }
        return 0;
    }
    max_gain(&root, &mut max_path);
    return max_path;
}

///p125
/// Given a string s, return true if it is a palindrome, or false otherwise.
///A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
pub fn is_palindrome(s: String) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let (mut left, mut right) = (0, chars.len() - 1);
    while left < right {
        while left < right && !chars[left].is_alphanumeric() {
            left = left + 1;
        }
        while left < right && !chars[right].is_alphanumeric() {
            right = right - 1;
        }
        if left < right {
            if chars[left].to_lowercase().cmp(chars[right].to_lowercase())
                != core::cmp::Ordering::Equal
            {
                return false;
            }
            left = left + 1;
            right = right - 1;
        }
    }
    return true;
}

///p126

///p127

///p128
///
/// Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
    let mut record = 0;
    let nums_set: std::collections::HashSet<&i32> =
        std::collections::HashSet::from_iter(nums.iter());
    for &item in nums_set.iter() {
        if let Some(_) = nums_set.get(&(item - 1)) {
            continue;
        } else {
            let mut this_record = 1;
            let mut target = item + 1;
            while let Some(_) = nums_set.get(&target) {
                target = target + 1;
                this_record = this_record + 1;
            }
            record = record.max(this_record);
        }
    }
    return record;
}

///p129
///
/// You are given the root of a binary tree containing digits from 0 to 9 only.
/// Each root-to-leaf path in the tree represents a number.
/// For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
/// Return the total sum of all root-to-leaf numbers.
pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn path(root: &Option<Rc<RefCell<TreeNode>>>) -> Vec<String> {
        if let Some(node) = root {
            let left_path = path(&node.borrow().left);
            let right_path = path(&node.borrow().right);
            let mut new_left: Vec<String> = left_path
                .into_iter()
                .map(|mut val| {
                    val.insert_str(0, &node.borrow().val.to_string());
                    return val;
                })
                .collect();
            let mut new_right: Vec<String> = right_path
                .into_iter()
                .map(|mut val| {
                    val.insert_str(0, &node.borrow().val.to_string());
                    return val;
                })
                .collect();
            new_left.append(&mut new_right);
            if new_left.len() == 0 {
                return vec![node.borrow().val.to_string()];
            } else {
                return new_left;
            }
        }
        return vec![];
    }
    let all_path = path(&root);
    return all_path
        .iter()
        .fold(0, |acc, cur| acc + cur.parse::<i32>().unwrap_or(0));
}

///p130
///
/// You are given an m x n matrix board containing letters 'X' and 'O', capture regions that are surrounded:
pub fn solve(board: &mut Vec<Vec<char>>) {
    let rows = board.len();
    let cols = board[0].len();
    let mut mark = vec![vec![false; cols]; rows];
    fn mark_neighbors(
        rows: i32,
        cols: i32,
        row: i32,
        col: i32,
        mark: &mut Vec<Vec<bool>>,
        board: &mut Vec<Vec<char>>,
    ) {
        let range: [i32; 2] = [-1, 1];
        for ver in range {
            let location = [row + ver, col];
            if location[0] < rows && location[0] > -1 && location[1] < cols && location[1] > -1 {
                if board[location[0] as usize][location[1] as usize] == 'O'
                    && mark[location[0] as usize][location[1] as usize] == false
                {
                    mark[location[0] as usize][location[1] as usize] = true;
                    mark_neighbors(rows, cols, location[0], location[1], mark, board)
                }
            }
        }
        for hor in range {
            let location = [row, col + hor];
            if location[0] < rows && location[0] > -1 && location[1] < cols && location[1] > -1 {
                if board[location[0] as usize][location[1] as usize] == 'O'
                    && mark[location[0] as usize][location[1] as usize] == false
                {
                    mark[location[0] as usize][location[1] as usize] = true;
                    mark_neighbors(rows, cols, location[0], location[1], mark, board)
                }
            }
        }
    }
    for row in vec![0, rows - 1] {
        for col in 0..cols {
            if board[row][col] == 'O' {
                mark[row][col] = true;
                mark_neighbors(
                    rows as i32,
                    cols as i32,
                    row as i32,
                    col as i32,
                    &mut mark,
                    board,
                )
            }
        }
    }
    for col in vec![0, cols - 1] {
        for row in 0..rows {
            if board[row][col] == 'O' {
                mark[row][col] = true;
                mark_neighbors(
                    rows as i32,
                    cols as i32,
                    row as i32,
                    col as i32,
                    &mut mark,
                    board,
                )
            }
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            if mark[r][c] == false && board[r][c] == 'O' {
                board[r][c] = 'X'
            }
        }
    }
}

///p131
///
/// Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
pub fn partition(s: String) -> Vec<Vec<String>> {
    let chars: Vec<char> = s.chars().collect();
    let length = chars.len();
    let mut ans: Vec<Vec<String>> = vec![];
    let mut current: Vec<String> = vec![];
    fn dfs(
        current: &mut Vec<String>,
        ans: &mut Vec<Vec<String>>,
        chars: &Vec<char>,
        length: usize,
        index: usize,
    ) {
        if index == length {
            ans.push(current.to_vec());
            return;
        } else {
            for j in index..length {
                let in_order = Vec::from_iter(chars[index..=j].iter().cloned());
                let rev_order = Vec::from_iter(chars[index..=j].iter().rev().cloned());
                if in_order == rev_order {
                    current.push(in_order.into_iter().collect::<String>());
                    dfs(current, ans, chars, length, j + 1);
                    current.pop();
                }
            }
        }
    }

    dfs(&mut current, &mut ans, &chars, length, 0);
    return ans;
}

///p132
///
/// Return the minimum cuts needed for a palindrome partitioning of s.
pub fn min_cut(s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let length = chars.len();
    let mut measure: Vec<Vec<bool>> = vec![vec![true; length]; length];
    for i in (0..length).rev() {
        for j in 0..length {
            if i >= j {
                measure[i][j] = true;
            } else {
                measure[i][j] = chars[i] == chars[j] && measure[i + 1][j - 1];
            }
        }
    }
    let mut cuts: Vec<usize> = vec![length; length];
    for i in 0..length {
        if measure[0][i] == true {
            cuts[i] = 0;
        } else {
            for k in 0..i {
                if measure[k + 1][i] == true {
                    cuts[i] = cuts[i].min(cuts[k] + 1);
                }
            }
        }
    }
    return cuts[length - 1] as i32;
}

///p133

///p134
///
/// Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.
pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
    let mut cur = 0;
    let (mut sum, mut pre) = (0, 0);
    let retain: Vec<i32> = gas.iter().zip(cost).map(|(&x, y)| x - y).collect();

    for (i, &n) in retain.iter().enumerate() {
        sum += n;
        if sum < 0 {
            pre += sum;
            sum = 0;
            cur = i + 1;
        }
    }
    if pre + sum < 0 {
        -1
    } else {
        cur as i32
    }
}

///p135
///
/// There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.
/// You are giving candies to these children subjected to the following requirements:
/// Each child must have at least one candy.
/// Children with a higher rating get more candies than their neighbors.
/// Return the minimum number of candies you need to have to distribute the candies to the children.
pub fn candy(ratings: Vec<i32>) -> i32 {
    let length = ratings.len();
    let mut left = vec![1; length];
    for i in 1..length {
        if ratings[i] > ratings[i - 1] {
            left[i] = left[i - 1] + 1;
        }
    }
    let mut right = vec![1; length];
    for i in (0..length - 1).rev() {
        if ratings[i] > ratings[i + 1] {
            right[i] = right[i + 1] + 1;
        }
    }
    return left
        .iter()
        .zip(right)
        .fold(0, |acc, cur| acc + cur.0.max(&cur.1));
}

///p136
pub fn single_number(nums: Vec<i32>) -> i32 {
    let mut record: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for i in nums {
        if let Some(_) = record.get(&i) {
            record.remove(&i);
        } else {
            record.insert(i);
        }
    }
    *record.iter().collect::<Vec<&i32>>()[0]
}

///p137
///
/// Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it.
/// You must implement a solution with a linear runtime complexity and use only constant extra space.
pub fn single_number_2(nums: Vec<i32>) -> i32 {
    let mut record: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for i in nums {
        match record.get(&i) {
            None => {
                record.insert(i, 1);
            }
            Some(1) => {
                record.insert(i, 2);
            }
            Some(2) => {
                record.remove(&i);
            }
            Some(_) => {}
        }
    }
    return record.iter().map(|item| *(item.0)).collect::<Vec<i32>>()[0];
}

///p138

///p139
///
///Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.
pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let mut dp = vec![false; chars.len() + 1];
    dp[0] = true;
    for i in 1..=chars.len() {
        for j in 0..i {
            if dp[j] == true && word_dict.contains(&s[j..i].to_string()) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[chars.len()];
}

///p140
///
/// Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
pub fn word_break_2(s: String, word_dict: Vec<String>) -> Vec<String> {
    let mut solutions = Vec::new();
    let mut solution = Vec::new();
    let g = word_dict
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    fn dfs(
        s: &str,
        g: &std::collections::HashSet<String>,
        solutions: &mut Vec<String>,
        solution: &mut Vec<String>,
        pos: usize,
    ) {
        let length = s.len();

        if pos == length {
            solutions.push(solution.join(" "));
            return;
        }

        for i in (pos + 1)..=length {
            let w = s[pos..i].to_string();

            if g.contains(&w) {
                solution.push(w);
                dfs(s, g, solutions, solution, i);
                solution.pop();
            }
        }
    }

    dfs(&s, &g, &mut solutions, &mut solution, 0);

    solutions
}

///p141

///p142

///p143  
///
///   
pub fn reorder_list(head: &mut Option<Box<ListNode>>) {
    let mut list: Vec<Box<ListNode>> = vec![];
    let mut pointer = head.clone();
    while let Some(mut node) = pointer {
        let next = node.next.take();
        list.push(node.clone());
        pointer = next;
    }

    let cnt = list.len();
    if cnt < 3 {
        return;
    }

    let mut end = None;
    if cnt & 1 == 1 {
        end = Some(list[cnt / 2].clone());
    }
    for i in (cnt + 1) / 2..cnt {
        let mut one = list[cnt - 1 - i].clone();
        let mut two = list[i].clone();
        two.next = end.take();

        one.next.replace(two);

        end.replace(one);
    }
    head.replace(end.unwrap());
}

///p144
///
///Given the root of a binary tree, return the preorder traversal of its nodes' values.
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ans = vec![];
    match root {
        None => {}
        Some(node) => {
            ans.push(node.borrow().val);
            let mut left = preorder_traversal(node.borrow().left.clone());
            let mut right = preorder_traversal(node.borrow().right.clone());
            ans.append(&mut left);
            ans.append(&mut right);
        }
    }
    return ans;
}

///p145
///
///Given the root of a binary tree, return the postorder traversal of its nodes' values.
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ans = vec![];
    match root {
        None => {}
        Some(node) => {
            let mut left = postorder_traversal(node.borrow_mut().left.take());
            let mut right = postorder_traversal(node.borrow_mut().right.take());
            ans.append(&mut left);
            ans.append(&mut right);
            ans.push(node.borrow().val);
        }
    }
    return ans;
}

///p146

///p147
///
pub fn insertion_sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut anchor = Some(Box::new(ListNode::new(0)));
    let mut head = head;

    while let Some(mut node) = head {
        head = node.next.take();

        let mut tail = &mut anchor;
        while tail.as_ref().unwrap().next.is_some()
            && (tail.as_ref().unwrap().next.as_ref().unwrap().val < node.val)
        {
            tail = &mut tail.as_mut().unwrap().next;
        }

        let half2 = tail.as_mut().unwrap().next.take();
        node.next = half2;
        tail.as_mut().unwrap().next = Some(node);
    }

    match anchor {
        Some(x) => x.next,
        None => None,
    }
}

///p148
///
/// Given the head of a linked list, return the list after sorting it in ascending order.
pub fn sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut values: Vec<i32> = vec![];
    let mut pointer = &head;
    while let Some(node) = pointer {
        values.push(node.val);
        pointer = &node.next;
    }
    values.sort();
    if values.len() == 0 {
        return None;
    } else {
        let mut head = Some(Box::new(ListNode::new(values[0])));
        let mut pointer = &mut head;
        for i in 1..values.len() {
            if let Some(node) = pointer {
                node.next = Some(Box::new(ListNode::new(values[i])));
                pointer = &mut node.next;
            }
        }
        return head;
    }
}

///p149
///
/// Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let c = a % b;
        a = b;
        b = c;
    }
    a
}
pub fn max_points(points: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashMap;
    let mut best = 0;
    for (i, p1) in points.iter().enumerate() {
        let (x1, y1) = (p1[0], p1[1]);

        // 记录一条直线出现的次数
        let mut cnts = HashMap::<(i32, i32, i32), i32>::new();
        // 从下标i + 1开始遍历
        for p2 in points.iter().skip(i + 1) {
            let (x2, y2) = (p2[0], p2[1]);
            let v3 = [y2 - y1, x1 - x2, x2 * y1 - x1 * y2];
            // 记录第一个非零元素的下标
            let mut nzi = -1;
            for k in 0..3 {
                if v3[k] != 0 {
                    nzi = k as i32;
                    break;
                }
            }
            let mut g = 1;
            if nzi != -1 {
                // 约分
                g = v3.iter().fold(v3[nzi as usize], |g, &x| {
                    // 不需要考虑0
                    if x == 0 {
                        g
                    } else {
                        gcd(g, x)
                    }
                });
                g = g.abs();
                if v3[nzi as usize] < 0 {
                    g = -g;
                }
            }
            // 进行约分并转为元组类型
            let v3 = (v3[0] / g, v3[1] / g, v3[2] / g);
            *cnts.entry(v3).or_insert(0) += 1;
            best = best.max(cnts[&v3]);
        }
    }
    best + 1
}

///p150
///
/// Evaluate the expression. Return an integer that represents the value of the expression.
///
/// Input: tokens = ["4","13","5","/","+"]
///
/// Output: 6
///
/// Explanation: (4 + (13 / 5)) = 6
pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    let mut number_stack: Vec<i32> = vec![];
    for token in tokens {
        if let Ok(number) = token.parse::<i32>() {
            number_stack.push(number);
        } else {
            let right = number_stack.pop().unwrap();
            let left = number_stack.pop().unwrap();
            number_stack.push(match token.as_str() {
                "+" => left + right,
                "-" => left - right,
                "*" => left * right,
                _ => left / right,
            })
        }
    }
    return number_stack[0];
}

/// p151
///
/// Return a string of the words in reverse order concatenated by a single space.
pub fn reverse_words(s: String) -> String {
    let mut words: Vec<&str> = s.split(' ').collect::<Vec<&str>>();
    words.reverse();
    return words
        .into_iter()
        .filter(|item| *item != "")
        .fold("".to_owned(), |mut acc, cur| {
            if acc != "" {
                acc.push_str(" ");
            }
            acc.push_str(cur);
            return acc;
        });
}

/// p152
///
/// Given an integer array nums, find a subarray that has the largest product, and return the product.
pub fn max_product(nums: Vec<i32>) -> i32 {
    let (mut mx, mut mn, mut ans) = (nums[0], nums[0], nums[0]);

    for i in 1..nums.len() {
        if nums[i] < 0 {
            std::mem::swap(&mut mx, &mut mn);
        }

        mx = nums[i].max(mx * nums[i]);
        mn = nums[i].min(mn * nums[i]);
        ans = ans.max(mx);
    }
    ans
}

/// p153
///
/// Given the sorted rotated array nums of unique elements, return the minimum element of this array.
pub fn find_min(nums: Vec<i32>) -> i32 {
    let mut low = 0;
    let mut high = nums.len() - 1;
    while low < high {
        let pivot = (high + low) / 2;
        if nums[pivot] < nums[high] {
            high = pivot;
        } else {
            low = pivot + 1;
        }
    }
    return nums[low];
}

/// p154
pub fn find_min_2(nums: Vec<i32>) -> i32 {
    return *nums.iter().min().unwrap();
}

///p155
struct MinStack {
    stk: Vec<i32>,
    min: Vec<i32>,
}

impl MinStack {
    fn new() -> Self {
        Self {
            stk: vec![],
            min: vec![],
        }
    }

    fn push(&mut self, val: i32) {
        self.stk.push(val);
        if self.min.is_empty() || val <= *self.min.last().unwrap() {
            self.min.push(val);
        }
    }

    fn pop(&mut self) {
        if self.stk.pop().unwrap() == *self.min.last().unwrap() {
            self.min.pop();
        }
    }

    fn top(&self) -> i32 {
        *self.stk.last().unwrap()
    }

    fn get_min(&self) -> i32 {
        *self.min.last().unwrap()
    }
}

///p162
///
/// Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
pub fn find_peak_element(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let mut l = 0;
    let mut r = n;
    while l != r {
        let mid = l + r >> 1;
        if mid + 1 != n && nums[mid] <= nums[mid + 1] {
            l = mid + 1; // discard
        } else {
            r = mid;
        }
    }
    l as i32
}
