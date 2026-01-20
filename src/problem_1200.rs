use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use crate::common::TreeNode;

/// 1103
pub fn distribute_candies(mut candies: i32, n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut ans = vec![0; n];
    let mut i = 1;
    while candies > 0 {
        ans[(i - 1) as usize % n] += i.min(candies);
        candies -= i;
        i += 1;
    }
    ans
}

/// 1104
pub fn path_in_zig_zag_tree(label: i32) -> Vec<i32> {
    let mut x = 2_i32.pow((label as f64).log2() as u32);
    let mut label = label;
    let mut ret = vec![label];

    while label > 1 {
        label = x - 1 - ((label / 2) % (x / 2));
        x /= 2;
        ret.push(label);
    }

    ret.reverse();
    ret
}

/// 1105
pub fn min_height_shelves(books: Vec<Vec<i32>>, shelf_width: i32) -> i32 {
    let n = books.len();
    let mut dp = vec![0; n + 1];
    for i in 1..=n {
        let (mut j, mut width, mut max_height) = (i - 1, books[i - 1][0], books[i - 1][1]);
        dp[i] = dp[i - 1] + max_height;
        while j > 0 && width + books[j - 1][0] <= shelf_width {
            max_height = max_height.max(books[j - 1][1]);
            width += books[j - 1][0];
            dp[i] = dp[i].min(dp[j - 1] + max_height);
            j -= 1;
        }
    }
    dp[n]
}

/// 1106
pub fn parse_bool_expr(expression: String) -> bool {
    let mut stack = VecDeque::new();
    expression.chars().for_each(|c| match c {
        '(' | 't' | 'f' | '!' | '&' | '|' => {
            stack.push_back(c);
        }
        ')' => {
            let mut list = vec![];
            while *stack.back().unwrap() != '(' {
                list.push(stack.pop_back().unwrap())
            }
            stack.pop_back();
            match stack.pop_back().unwrap() {
                '&' => {
                    if list.contains(&'f') {
                        stack.push_back('f')
                    } else {
                        stack.push_back('t')
                    }
                }
                '!' => {
                    if list.contains(&'f') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                '|' => {
                    if list.contains(&'t') {
                        stack.push_back('t')
                    } else {
                        stack.push_back('f')
                    }
                }
                _ => (),
            }
        }
        ',' => (),
        _ => (),
    });
    stack.pop_back().unwrap() == 't'
}

/// 1108
pub fn defang_i_paddr(address: String) -> String {
    address.replace(".", "[.]")
}

/// 1109
pub fn corp_flight_bookings(bookings: Vec<Vec<i32>>, n: i32) -> Vec<i32> {
    let (mut get_on, mut get_off) = (HashMap::new(), HashMap::new());
    bookings.iter().for_each(|b| {
        let on = get_on.entry(b[0]).or_insert(0);
        *on = *on + b[2];
        let off = get_off.entry(b[1] + 1).or_insert(0);
        *off = *off + b[2];
    });
    let (mut total, mut time_line) = (0, vec![0; n as usize]);
    time_line.iter_mut().enumerate().for_each(|(day, cnt)| {
        let (on, off) = (
            get_on.get(&((day + 1) as i32)).unwrap_or(&0),
            get_off.get(&((day + 1) as i32)).unwrap_or(&0),
        );
        total = total + on - off;
        *cnt = total;
    });
    time_line
}

// /// 1110
// pub fn del_nodes(
//     root: Option<Rc<RefCell<TreeNode>>>,
//     mut to_delete: Vec<i32>,
// ) -> Vec<Option<Rc<RefCell<TreeNode>>>> {
//     let mut trees = vec![root];
//     fn delete_target(
//         target: i32,
//         tree: Option<Rc<RefCell<TreeNode>>>,
//     ) -> (bool, Vec<Option<Rc<RefCell<TreeNode>>>>) {
//         let mut new_tree = vec![];
//         if let Some(node) = tree.as_ref() {
//             if node.borrow().val == target {
//                 if node.borrow().left.is_some() {
//                     new_tree.push(node.borrow().left.clone());
//                 }
//                 if node.borrow().right.is_some() {
//                     new_tree.push(node.borrow().right.clone());
//                 }
//                 node.borrow_mut().left = None;
//                 node.borrow_mut().right = None;
//                 return (true, new_tree);
//             } else {
//                 let (left_removed, mut left_tree) =
//                     delete_target(target, node.borrow().left.clone());
//                 let (right_memoved, mut right_tree) =
//                     delete_target(target, node.borrow().right.clone());
//                 if left_removed {
//                     node.borrow_mut().left = None;
//                 }
//                 if right_memoved {
//                     node.borrow_mut().right = None;
//                 }
//                 new_tree.push(tree);
//                 new_tree.append(&mut left_tree);
//                 new_tree.append(&mut right_tree);
//             }
//         }
//         (false, new_tree)
//     }
//     while to_delete.len() != 0 {
//         let target = to_delete.pop().unwrap();
//         let size = trees.len();
//         for _ in 0..size {
//             let (_, mut new_tree) = delete_target(target, trees.remove(0));
//             trees.append(&mut new_tree);
//         }
//     }
//     trees
// }
