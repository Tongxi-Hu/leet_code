use std::{cell::RefCell, collections::HashSet, rc::Rc};

use crate::common::{ListNode, TreeNode};

///p201
pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    let m = 32 - (left ^ right).leading_zeros();
    left & !((1 << m) - 1)
}

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

///p203
pub fn remove_elements(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut fake = Box::new(ListNode { val: 0, next: head });
    let mut pointer: &mut Box<ListNode> = &mut fake;
    while let Some(ref mut node) = pointer.next {
        if node.val == val {
            pointer.next = node.next.take();
        } else {
            pointer = pointer.next.as_mut().unwrap();
        }
    }
    return fake.next;
}

///p204
pub fn count_primes(n: i32) -> i32 {
    let n = n as usize;
    let mut d = vec![true; n];
    let mut count = 0;
    for i in 2..n {
        if d[i] {
            count += 1;
            let mut j = i * i;
            while j < n {
                d[j] = false;
                j += i;
            }
        }
    }
    count
}

/// p205
pub fn is_isomorphic(s: String, t: String) -> bool {
    let s_char = s.chars().collect::<Vec<char>>();
    let t_char = t.chars().collect::<Vec<char>>();
    if s_char.len() != t_char.len() {
        return false;
    };
    let mut s_to_t: std::collections::HashMap<char, char> = std::collections::HashMap::new();
    let mut t_to_s: std::collections::HashMap<char, char> = std::collections::HashMap::new();
    for i in 0..s_char.len() {
        match (s_to_t.get(&s_char[i]), t_to_s.get(&t_char[i])) {
            (Some(t), Some(s)) => {
                if *t == t_char[i] && *s == s_char[i] {
                    continue;
                } else {
                    return false;
                }
            }
            (None, Some(_)) | (Some(_), None) => {
                return false;
            }
            (None, None) => {
                s_to_t.insert(s_char[i], t_char[i]);
                t_to_s.insert(t_char[i], s_char[i]);
                continue;
            }
        }
    }
    return true;
}

/// p206
pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut pre = None;
    let mut cur = head;
    while let Some(mut node) = cur {
        let next = node.next;
        node.next = pre;
        pre = Some(node);
        cur = next;
    }
    return pre;
}
