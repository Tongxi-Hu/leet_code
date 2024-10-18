use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::common::{ListNode, TreeNode};

///p202
pub fn is_happy(n: i32) -> bool {
    let mut n = n;
    let mut history: std::collections::HashSet<i32> = std::collections::HashSet::new();
    while n != 1 {
        n = n
            .to_string()
            .chars()
            .map(|char| char.to_digit(10).unwrap())
            .fold(0, |acc, cur| return acc + (cur * cur) as i32);
        if n == 1 {
            return true;
        };
        match history.get(&n) {
            Some(_) => {
                return false;
            }
            None => {
                history.insert(n);
            }
        }
    }
    return true;
}
