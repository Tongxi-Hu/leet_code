use std::collections::HashMap;

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
