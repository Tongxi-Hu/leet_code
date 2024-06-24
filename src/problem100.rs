use std::collections::{HashMap, HashSet};

///p1 two sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut register: HashMap<i32, i32> = HashMap::new();
    let mut location: Vec<i32> = vec![];
    for (index, val) in nums.iter().enumerate() {
        match register.get(&(target - val)) {
            None => {
                register.insert(*val, index as i32);
            }
            Some(i) => {
                location.push(*i);
                location.push(index as i32);
            }
        }
    }
    return location;
}

#[test]
pub fn test_two_sum() {
    assert_eq!(vec![0, 1], two_sum(vec![2, 7, 11, 15], 9));
    assert_eq!(vec![1, 2], two_sum(vec![3, 2, 4], 6));
}

///p2 add number
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    if l1 == None && l2 == None {
        return None;
    }
    let mut start: Option<Box<ListNode>> = None;
    let mut p1 = &l1;
    let mut p2 = &l2;
    let mut p = &mut start;
    let mut acc: i32 = 0;
    loop {
        match (p1, p2) {
            (Some(val1), Some(val2)) => {
                let mut sum = val1.val + val2.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p1 = &val1.next;
                p2 = &val2.next;
            }
            (None, Some(val2)) => {
                let mut sum = val2.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p2 = &val2.next;
            }
            (Some(val1), None) => {
                let mut sum = val1.val + acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                p1 = &val1.next;
            }
            (None, None) => {
                let mut sum = acc;
                if sum >= 10 {
                    let temp = sum - 10;
                    acc = (sum - temp) / 10;
                    sum = temp;
                } else {
                    acc = 0;
                }
                if (sum == 0) {
                    break;
                }
                *p = Some(Box::new(ListNode::new(sum)));
                p = &mut (p.as_mut().unwrap().next);
                break;
            }
        }
    }
    return start;
}

#[test]
pub fn test_add_two_numbers() {
    assert_eq!(
        Some(Box::new(ListNode {
            val: 7,
            next: Some(Box::new(ListNode { val: 1, next: None }))
        })),
        add_two_numbers(
            Some(Box::new(ListNode { val: 8, next: None })),
            Some(Box::new(ListNode { val: 9, next: None }))
        )
    );
}

///p3 longest substring
pub fn length_of_longest_substring(s: String) -> i32 {
    let mut left = 0;
    let mut right = 0;
    let mut max = 0;
    let mut collection = HashSet::<char>::new();
    let char_vec = s.chars().collect::<Vec<char>>();
    while right < char_vec.len() {
        match char_vec.get(right) {
            None => (),
            Some(char) => match collection.get(char) {
                None => {
                    collection.insert(*char);
                    max = collection.len().max(max);
                    right = right + 1;
                }
                Some(_) => {
                    while collection.get(char).is_some() {
                        match char_vec.get(left) {
                            None => (),
                            Some(val) => {
                                collection.remove(&val);
                            }
                        }
                        left = left + 1;
                    }
                }
            },
        }
    }
    return max as i32;
}

#[test]
fn test_length_of_longest_substring() {
    assert_eq!(3, length_of_longest_substring("abcabcbb".to_string()));
    assert_eq!(3, length_of_longest_substring("pwwkew".to_string()));
}

///p4 median of two sorted arrays
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let mut combine: Vec<i32> = Vec::new();
    let len = nums1.len() + nums2.len();
    let mut p1 = 0;
    let mut p2 = 0;
    for _ in 0..len {
        match (nums1.get(p1), nums2.get(p2)) {
            (Some(val1), Some(val2)) => {
                if val1 <= val2 {
                    combine.push(*val1);
                    p1 = p1 + 1;
                } else {
                    combine.push(*val2);
                    p2 = p2 + 1;
                }
            }
            (Some(val1), None) => {
                combine.push(*val1);
                p1 = p1 + 1;
            }
            (None, Some(val2)) => {
                combine.push(*val2);
                p2 = p2 + 1;
            }
            (None, None) => {}
        }
    }

    if len % 2 == 0 {
        return (combine[len / 2 - 1] as f64 + combine[len / 2] as f64) / 2.0;
    } else {
        return combine[len / 2] as f64;
    }
}

#[test]
fn test_find_median_sorted_arrays() {
    assert_eq!(2.0, find_median_sorted_arrays(vec![1, 3], vec![2]));
    assert_eq!(2.5, find_median_sorted_arrays(vec![1, 2, 3, 4], vec![]));
}

///p7
pub fn reverse(x: i32) -> i32 {
    let mut x = x;
    let mut result = 0;
    while x != 0 {
        if result > i32::MAX / 10 || result < i32::MIN / 10 {
            return 0;
        }
        result = result * 10 + x % 10;
        x = x / 10;
    }
    return result;
}

#[test]
fn test_reverse() {
    assert_eq!(-321, reverse(-123));
    assert_eq!(21, reverse(120));
}

///p8
pub fn my_atoi(s: String) -> i32 {
    let mut sign = '+';
    let mut val: i64 = 0;
    let mut has_content = false;
    for char in s.chars().into_iter() {
        match char.to_digit(10) {
            Some(digit) => {
                val = val * 10 + (digit as i64);
                has_content = true;
                if sign == '-' {
                    if -val < i32::MIN as i64 {
                        return i32::MIN;
                    }
                } else {
                    if val > i32::MAX as i64 {
                        return i32::MAX;
                    }
                }
            }
            None => {
                if char.is_whitespace() {
                    if has_content {
                        break;
                    } else {
                        continue;
                    }
                } else if char == '-' || char == '+' {
                    if has_content {
                        break;
                    } else {
                        has_content = true;
                        sign = char;
                    }
                } else {
                    break;
                }
            }
        }
    }
    if sign == '-' {
        return (val as i32) * -1;
    } else {
        return val as i32;
    }
}

#[test]
fn test_my_atoi() {
    assert_eq!(-42, my_atoi(" -042".to_string()));
    assert_eq!(1337, my_atoi("1337c0d3".to_string()));
}
