use std::{cell::RefCell, cmp::Reverse, collections::BinaryHeap, rc::Rc};

use crate::common::TreeNode;

/// p701
pub fn insert_into_bst(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    let mut root = root;
    if let Some(node) = root.as_ref() {
        let mut tree = node.borrow_mut();
        let v = tree.val;
        if v > val {
            tree.left = insert_into_bst(tree.left.clone(), val);
        } else {
            tree.right = insert_into_bst(tree.right.clone(), val)
        }
    } else {
        root = Some(Rc::new(RefCell::new(TreeNode::new(val))))
    }
    root
}

/// p703
struct KthLargest {
    heap: BinaryHeap<Reverse<i32>>,
}

impl KthLargest {
    fn new(k: i32, nums: Vec<i32>) -> Self {
        let mut kth = Self {
            heap: BinaryHeap::with_capacity(k as usize),
        };
        nums.iter().for_each(|&n| kth.push(n));
        kth
    }

    fn add(&mut self, val: i32) -> i32 {
        self.push(val);
        self.peek()
    }

    fn push(&mut self, n: i32) {
        if self.heap.len() == self.heap.capacity() {
            if *self.heap.peek().unwrap() > Reverse(n) {
                self.heap.pop();
            } else {
                return;
            }
        }
        self.heap.push(Reverse(n));
    }

    fn peek(&self) -> i32 {
        self.heap.peek().unwrap().0
    }
}

/// p704
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let (mut l, mut r) = (0, nums.len());
    while l < r {
        let mid = l + (r - l) / 2;
        if nums[mid] == target {
            return mid as i32;
        } else if nums[mid] > target {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    -1
}

/// p705
struct MyHashSet {
    arr: Vec<bool>,
}

impl MyHashSet {
    fn new() -> Self {
        Self {
            arr: vec![false; 1000001],
        }
    }

    fn add(&mut self, key: i32) {
        self.arr[key as usize] = true;
    }

    fn remove(&mut self, key: i32) {
        if !self.arr[key as usize] {
            return;
        }
        self.arr[key as usize] = false;
    }

    fn contains(&self, key: i32) -> bool {
        self.arr[key as usize]
    }
}

/// p706
struct MyHashMap {
    nums: Vec<i32>,
}

impl MyHashMap {
    fn new() -> Self {
        Self {
            nums: vec![-1; 1000001],
        }
    }

    fn put(&mut self, key: i32, value: i32) {
        self.nums[key as usize] = value;
    }

    fn get(&self, key: i32) -> i32 {
        self.nums[key as usize]
    }

    fn remove(&mut self, key: i32) {
        self.nums[key as usize] = -1;
    }
}

/// p707
struct ListNode {
    prev: Option<Rc<RefCell<ListNode>>>,
    next: Option<Rc<RefCell<ListNode>>>,
    val: i32,
}

impl ListNode {
    fn new(
        val: i32,
        prev: Option<Rc<RefCell<ListNode>>>,
        next: Option<Rc<RefCell<ListNode>>>,
    ) -> ListNode {
        ListNode { prev, next, val }
    }
}

struct MyLinkedList {
    head: Option<Rc<RefCell<ListNode>>>,
    size: i32,
}

impl MyLinkedList {
    fn new() -> Self {
        let head = Rc::new(RefCell::new(ListNode::new(-1, None, None)));
        head.borrow_mut().prev = Some(head.clone());
        head.borrow_mut().next = Some(head.clone());
        MyLinkedList {
            head: Some(head),
            size: 0,
        }
    }

    fn get(&self, index: i32) -> i32 {
        if index < 0 || index >= self.size {
            return -1;
        }
        self.find_node(index).as_ref().unwrap().borrow().val
    }

    fn add_at_head(&mut self, val: i32) {
        let next = self.head.as_ref().unwrap().borrow().next.clone();
        let curr = Rc::new(RefCell::new(ListNode::new(
            val,
            self.head.clone(),
            next.clone(),
        )));
        next.clone().as_ref().unwrap().borrow_mut().prev = Some(curr.clone());
        self.head.as_ref().unwrap().borrow_mut().next = Some(curr.clone());
        self.size += 1;
    }

    fn add_at_tail(&mut self, val: i32) {
        let prev = self.head.as_ref().unwrap().borrow().prev.clone();
        let curr = Rc::new(RefCell::new(ListNode::new(
            val,
            prev.clone(),
            self.head.clone(),
        )));
        prev.clone().as_ref().unwrap().borrow_mut().next = Some(curr.clone());
        self.head.as_ref().unwrap().borrow_mut().prev = Some(curr.clone());
        self.size += 1;
    }

    fn add_at_index(&mut self, index: i32, val: i32) {
        if index > self.size {
            return;
        }
        if index < 0 {
            self.add_at_head(val);
            return;
        }
        if index == self.size {
            self.add_at_tail(val);
            return;
        }
        let curr = self.find_node(index);
        let new_node = Rc::new(RefCell::new(ListNode::new(
            val,
            curr.as_ref().unwrap().borrow().prev.clone(),
            curr.clone(),
        )));
        curr.as_ref()
            .unwrap()
            .borrow_mut()
            .prev
            .as_ref()
            .unwrap()
            .borrow_mut()
            .next = Some(new_node.clone());
        curr.as_ref().unwrap().borrow_mut().prev = Some(new_node.clone());
        self.size += 1;
    }

    fn delete_at_index(&mut self, index: i32) {
        if self.size <= 0 || index < 0 || index >= self.size {
            return;
        }
        if self.size == 1 {
            self.head.as_ref().unwrap().borrow_mut().prev = self.head.clone();
            self.head.as_ref().unwrap().borrow_mut().next = self.head.clone();
            self.size -= 1;
        }
        let curr = self.find_node(index);
        let (prev, next) = (
            curr.as_ref().unwrap().borrow().prev.clone(),
            curr.as_ref().unwrap().borrow().next.clone(),
        );
        prev.clone().as_ref().unwrap().borrow_mut().next = next.clone();
        next.clone().as_ref().unwrap().borrow_mut().prev = prev.clone();
        self.size -= 1;
    }

    fn find_node(&self, index: i32) -> Option<Rc<RefCell<ListNode>>> {
        if self.size == 0 {
            return self.head.as_ref().unwrap().borrow().next.clone();
        }
        if self.size == index {
            return self.head.as_ref().unwrap().borrow().prev.clone();
        }
        let mut curr = self.head.as_ref().unwrap().borrow().next.clone();
        for _ in 0..index {
            let node = curr.as_ref().unwrap().borrow().next.clone();
            curr = node;
        }
        return curr;
    }
}

/// p709
pub fn to_lower_case(s: String) -> String {
    s.to_lowercase()
}

/// p712
pub fn minimum_delete_sum(s1: String, s2: String) -> i32 {
    let s1 = s1.bytes().collect::<Vec<_>>();
    let s2 = s2.bytes().collect::<Vec<_>>();
    let mut dp = vec![0; s2.len() + 1];
    s1.iter().for_each(|&c| {
        let mut pre = 0;
        for j in 1..=s2.len() {
            let tmp = dp[j];
            if c == s2[j - 1] {
                dp[j] = pre + c as i32;
            } else {
                dp[j] = dp[j].max(dp[j - 1]);
            }
            pre = tmp;
        }
    });
    let sum1: i32 = s1.iter().map(|&x| x as i32).sum();
    let sum2: i32 = s2.iter().map(|&x| x as i32).sum();
    sum1 + sum2 - 2 * dp[s2.len()]
}
